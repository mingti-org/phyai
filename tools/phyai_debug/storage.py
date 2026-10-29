"""Nonblocking capture with a separate, crash-readable SQLite/PNG writer."""

from __future__ import annotations

import copy
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sqlite3
import threading
import time
from typing import Any

import numpy as np


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def json_value(value: Any) -> Any:
    """Keep invalid numeric evidence visible without emitting invalid JSON."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def writer_main(output, events, images, ready, failed, started_ns):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    database = None
    result = {"events": 0, "images": 0, "blobs": 0, "counts": {}, "error": None}
    blob_handles = {}
    try:
        database = sqlite3.connect(str(output / "trace.sqlite"))
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("PRAGMA synchronous=NORMAL")
        database.executescript(
            "CREATE TABLE events (seq INTEGER PRIMARY KEY, t_ns INTEGER, t REAL, "
            "kind TEXT, request_id TEXT, data TEXT NOT NULL);"
            "CREATE INDEX event_time ON events(t,seq);"
            "CREATE INDEX event_request ON events(request_id,seq);"
            "CREATE INDEX event_kind ON events(kind,t);"
            "CREATE TABLE arrays (seq INTEGER, name TEXT, dtype TEXT, shape TEXT, "
            "data BLOB, PRIMARY KEY(seq,name));"
        )
        database.commit()
        ready.set()
        event_end = image_end = False
        last_commit = time.monotonic()
        while not (event_end and image_end):
            if not event_end:
                try:
                    item = events.get(timeout=0.02)
                except queue.Empty:
                    item = False
                if item is None:
                    event_end = True
                elif item is not False:
                    row, arrays = item
                    row["t"] = (row["t_ns"] - started_ns) / 1e9
                    row["values"] = {}
                    row["array_specs"] = {}
                    for name, array in arrays.items():
                        row["values"][name] = json_value(array.tolist())
                        row["array_specs"][name] = {
                            "dtype": array.dtype.str,
                            "shape": list(array.shape),
                        }
                        database.execute(
                            "INSERT INTO arrays VALUES (?,?,?,?,?)",
                            (
                                row["seq"],
                                name,
                                array.dtype.str,
                                json.dumps(list(array.shape)),
                                array.tobytes(),
                            ),
                        )
                    encoded = json.dumps(
                        json_value(row), separators=(",", ":"), allow_nan=False
                    )
                    database.execute(
                        "INSERT INTO events VALUES (?,?,?,?,?,?)",
                        (
                            row["seq"],
                            row["t_ns"],
                            row["t"],
                            row["kind"],
                            str(row["request_id"])
                            if row.get("request_id") is not None
                            else None,
                            encoded,
                        ),
                    )
                    result["events"] += 1
                    counts = result["counts"]
                    counts[row["kind"]] = counts.get(row["kind"], 0) + 1
            if not image_end:
                try:
                    item = images.get_nowait()
                except queue.Empty:
                    item = False
                if item is None:
                    image_end = True
                elif item is not False:
                    from PIL import Image

                    image_items, blob_items = item
                    for relative, pixels in image_items:
                        path = output / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_suffix(".tmp")
                        Image.fromarray(pixels).save(temporary, format="PNG")
                        temporary.replace(path)
                        result["images"] += 1
                    for relative, offset, payload in blob_items:
                        if relative not in blob_handles:
                            path = output / relative
                            path.parent.mkdir(parents=True, exist_ok=True)
                            blob_handles[relative] = path.open("wb")
                        stream = blob_handles[relative]
                        if stream.tell() != offset:
                            raise RuntimeError("Compressed packet offset mismatch")
                        stream.write(payload)
                        result["blobs"] += 1
            if time.monotonic() - last_commit >= 0.25:
                database.commit()
                last_commit = time.monotonic()
        database.commit()
        database.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except BaseException as error:
        result["error"] = repr(error)
        failed.set()
    finally:
        for stream in blob_handles.values():
            stream.close()
        if database is not None:
            database.close()
        try:
            atomic_json(output / "writer_result.json", result)
        except OSError:
            failed.set()


class TraceRecorder:
    """Capture errors mark the trace incomplete; they never throw into control."""

    def __init__(
        self,
        output: Path,
        metadata: dict,
        queue_capacity: int = 4096,
        image_capacity: int = 64,
    ):
        if queue_capacity < 1 or image_capacity < 1:
            raise ValueError("Queue capacities must be positive")
        self.output = Path(output).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.started_ns = self.start_monotonic_ns = time.monotonic_ns()
        self.wall_started_ns = time.time_ns()
        self.metadata = copy.deepcopy(metadata)
        self.lock = threading.Lock()
        self.sequence = 0
        self.dropped_events = self.dropped_images = 0
        self.blob_offsets = {}
        self.dropped_media = 0
        self.capture_errors: list[str] = []
        self.closed = False
        self.manifest = {
            "format": "phyai-debug-v1",
            "complete": False,
            "status": "recording",
            "start_monotonic_ns": self.started_ns,
            "start_unix_ns": self.wall_started_ns,
            "metadata": self.metadata,
            "clock": "client host monotonic_ns",
        }
        atomic_json(self.output / "manifest.json", self.manifest)
        context = mp.get_context("spawn")
        self.events = context.Queue(maxsize=queue_capacity)
        self.images = context.Queue(maxsize=image_capacity)
        self.ready = context.Event()
        self.failed = context.Event()
        self.process = context.Process(
            target=writer_main,
            args=(
                self.output,
                self.events,
                self.images,
                self.ready,
                self.failed,
                self.started_ns,
            ),
            name="phyai-trace-writer",
            daemon=True,
        )
        self.process.start()
        deadline = time.monotonic() + 10
        while not self.ready.wait(0.05):
            if (
                not self.process.is_alive()
                or self.failed.is_set()
                or time.monotonic() >= deadline
            ):
                self.close(grace_s=1)
                raise RuntimeError(
                    "Trace writer could not initialize before client startup"
                )

    def emit(
        self,
        kind: str,
        *,
        t_ns: int | None = None,
        arrays: dict[str, np.ndarray] | None = None,
        images: dict[str, np.ndarray] | None = None,
        blobs: dict[str, bytes] | None = None,
        **fields,
    ) -> int:
        with self.lock:
            self.sequence += 1
            sequence = self.sequence
            if self.closed or self.failed.is_set() or not self.process.is_alive():
                self.dropped_events += 1
                return sequence
            try:
                row = copy.deepcopy(fields)
                row.update(
                    seq=sequence,
                    kind=kind,
                    t_ns=time.monotonic_ns() if t_ns is None else int(t_ns),
                )
                snapshots = {}
                for name, value in (arrays or {}).items():
                    value = np.asarray(value)
                    if value.dtype.hasobject:
                        raise ValueError("Object arrays are not trace data")
                    snapshots[str(name)] = np.array(value, copy=True, order="C")
                image_items = []
                blob_items = []
                row["images"] = {}
                for name, pixels in (images or {}).items():
                    if not name or any(
                        c
                        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                        for c in name
                    ):
                        raise ValueError("Invalid camera name")
                    pixels = np.asarray(pixels)
                    if (
                        pixels.dtype != np.uint8
                        or pixels.ndim != 3
                        or pixels.shape[-1] != 3
                    ):
                        raise ValueError("Model images must be HWC uint8 RGB")
                    relative = f"model_inputs/{sequence:09d}/{name}.png"
                    row["images"][name] = relative
                    image_items.append((relative, pixels.copy()))
                row["blobs"] = {}
                for name, payload in (blobs or {}).items():
                    if Path(name).name != name or not name.endswith(".h264"):
                        raise ValueError("Invalid compressed video name")
                    relative = "video/" + name
                    offset = self.blob_offsets.get(relative, 0)
                    blob_items.append((relative, offset, bytes(payload)))
                    row["blobs"][name] = {
                        "file": relative,
                        "offset": offset,
                        "size": len(payload),
                    }
                if image_items or blob_items:
                    try:
                        size = sum(p.nbytes for _, p in image_items) + sum(
                            len(p) for _, _, p in blob_items
                        )
                        if size > 32 * 1024**2:
                            raise queue.Full
                        self.images.put_nowait((image_items, blob_items))
                        for relative, offset, payload in blob_items:
                            self.blob_offsets[relative] = offset + len(payload)
                    except queue.Full:
                        self.dropped_images += len(image_items)
                        self.dropped_media += len(blob_items)
                        row.update(media_missing=True, images={}, blobs={})
                self.events.put_nowait((row, snapshots))
            except queue.Full:
                self.dropped_events += 1
            except Exception as error:
                self.dropped_events += 1
                if len(self.capture_errors) < 20:
                    self.capture_errors.append(repr(error))
            return sequence

    def mark_incomplete(self, error):
        with self.lock:
            if len(self.capture_errors) < 20:
                self.capture_errors.append(str(error))

    def close(self, grace_s: float = 15) -> dict:
        with self.lock:
            if self.closed:
                return self.manifest
            self.closed = True
        deadline = time.monotonic() + max(0, grace_s)
        errors = list(self.capture_errors)
        for channel in (self.events, self.images):
            while self.process.is_alive() and not self.failed.is_set():
                try:
                    channel.put(
                        None, timeout=min(0.05, max(0, deadline - time.monotonic()))
                    )
                    break
                except queue.Full:
                    if time.monotonic() >= deadline:
                        errors.append("Writer flush deadline exceeded")
                        break
        self.process.join(timeout=max(0, deadline - time.monotonic()))
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
            errors.append("Writer terminated before flush completed")
        for channel in (self.events, self.images):
            channel.cancel_join_thread()
            channel.close()
        result_path = self.output / "writer_result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if result.get("error"):
            errors.append(result["error"])
        if not result or self.process.exitcode != 0 or self.failed.is_set():
            errors.append("Writer exited unsuccessfully")
        complete = (
            not errors
            and not self.dropped_events
            and not self.dropped_images
            and not self.dropped_media
        )
        self.manifest.update(
            complete=complete,
            status="complete" if complete else "incomplete",
            duration_s=(time.monotonic_ns() - self.started_ns) / 1e9,
            counts=result.get("counts", {}),
            event_count=result.get("events", 0),
            image_count=result.get("images", 0),
            attempted_events=self.sequence,
            dropped_events=self.dropped_events,
            dropped_images=self.dropped_images,
            dropped_media=self.dropped_media,
            blob_count=result.get("blobs", 0),
            errors=errors,
        )
        try:
            atomic_json(self.output / "manifest.json", self.manifest)
        except OSError as error:
            self.manifest.update(complete=False, status="incomplete")
            self.manifest["errors"].append(repr(error))
        return self.manifest
