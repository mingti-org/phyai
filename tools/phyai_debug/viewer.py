"""Read-only, bounded queries over a trace; no ROS or policy imports."""

from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import sqlite3
from urllib.parse import parse_qs, unquote, urlsplit


class TraceReader:
    def __init__(self, trace):
        self.root = Path(trace).expanduser().resolve()
        if (
            not (self.root / "manifest.json").is_file()
            or not (self.root / "trace.sqlite").is_file()
        ):
            raise ValueError("Not a recorded trace directory")

    def connect(self):
        return sqlite3.connect(
            (self.root / "trace.sqlite").as_uri() + "?mode=ro", uri=True
        )

    def manifest(self):
        result = json.loads((self.root / "manifest.json").read_text())
        with self.connect() as db:
            low, high, count = db.execute(
                "SELECT MIN(t), MAX(t), COUNT(*) FROM events"
            ).fetchone()
        result.update(
            bounds=[low or 0, high or 0], duration=high or 0, event_count=count
        )
        result["robot_available"] = (self.root / "robot/robot.urdf").is_file()
        return result

    def events(
        self, *, start=0, end=1e12, limit=10000, after=0, kinds=(), request_id=None
    ):
        limit = max(1, min(int(limit), 50000))
        if (
            not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(end) < float(start)
        ):
            raise ValueError("Invalid time range")
        if len(kinds) > 32:
            raise ValueError("Too many event kinds")
        clauses, args = (
            ["t>=?", "t<=?", "seq>?"],
            [float(start), float(end), int(after)],
        )
        if kinds:
            clauses.append("kind IN (" + ",".join("?" for _ in kinds) + ")")
            args.extend(kinds)
        if request_id is not None:
            clauses.append("request_id=?")
            args.append(str(request_id))
        with self.connect() as db:
            rows = db.execute(
                "SELECT seq,data FROM events WHERE "
                + " AND ".join(clauses)
                + " ORDER BY seq LIMIT ?",
                [*args, limit + 1],
            ).fetchall()
        truncated = len(rows) > limit
        rows = rows[:limit]
        return {
            "events": [json.loads(row[1]) for row in rows],
            "truncated": truncated,
            "next_after": rows[-1][0] if truncated else None,
        }

    def timeline(self, after=0):
        result = self.events(
            limit=5000,
            after=after,
            kinds=("request", "response", "chunk", "event", "mode"),
        )
        for row in result["events"]:
            row.pop("values", None)
            row.pop("array_specs", None)
        result["bounds"] = self.manifest()["bounds"]
        return result


def safe_path(root, relative):
    relative = Path(unquote(relative))
    if relative.is_absolute() or ".." in relative.parts or "\x00" in str(relative):
        raise ValueError("Invalid asset path")
    path = root / relative
    if any(
        parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent
    ):
        raise ValueError("Symlink assets are not served")
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError(relative)
    return path


class TraceHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, reader, **kwargs):
        self.reader = reader
        super().__init__(*args, **kwargs)

    def do_GET(self):
        try:
            url = urlsplit(self.path)
            query = {key: values[-1] for key, values in parse_qs(url.query).items()}
            if url.path == "/api/manifest":
                return self.send_json(self.reader.manifest())
            if url.path == "/api/timeline":
                return self.send_json(self.reader.timeline(int(query.get("after", 0))))
            if url.path == "/api/request":
                if "id" not in query:
                    raise ValueError("Missing request id")
                return self.send_json(
                    self.reader.events(
                        request_id=query["id"],
                        kinds=("request", "response", "adapted_actions", "chunk"),
                    )
                )
            if url.path == "/api/events":
                return self.send_json(
                    self.reader.events(
                        start=float(query.get("start", 0)),
                        end=float(query.get("end", 1e12)),
                        after=int(query.get("after", 0)),
                        limit=int(query.get("limit", 10000)),
                        kinds=tuple(filter(None, query.get("kinds", "").split(","))),
                    )
                )
            if url.path.startswith("/data/"):
                root, relative = self.reader.root, url.path.removeprefix("/data/")
                if not Path(relative).parts or Path(relative).parts[0] not in (
                    "model_inputs",
                    "video",
                    "robot",
                ):
                    raise ValueError("Only recorded media can be downloaded")
            elif url.path.startswith("/robot/"):
                root, relative = (
                    self.reader.root / "robot",
                    url.path.removeprefix("/robot/"),
                )
            else:
                root = Path(__file__).with_name("web") / "dist"
                relative = "index.html" if url.path == "/" else url.path.lstrip("/")
            path = safe_path(root, relative)
            size = path.stat().st_size
            self.send_response(200)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            )
            self.send_header("Content-Length", str(size))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    self.wfile.write(block)
        except (ValueError, KeyError) as error:
            self.send_json({"error": str(error)}, 400)
        except FileNotFoundError:
            self.send_json(
                {
                    "error": "Asset missing; build the viewer or select a trace with robot assets"
                },
                404,
            )
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, value, status=200):
        payload = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def make_server(trace, host="127.0.0.1", port=8770):
    return ThreadingHTTPServer(
        (host, port), partial(TraceHandler, reader=TraceReader(trace))
    )


def serve_trace(trace, host="127.0.0.1", port=8770):
    with make_server(trace, host, port) as server:
        print(f"Read-only trace viewer: http://{host}:{server.server_port}", flush=True)
        server.serve_forever()
