# MiniCPM-RobotTrack in PhyAI

The plugin supports the complete model path:

```text
32 RGB frames + instruction
  -> PIL-BICUBIC resize and tokenization
  -> DINOv3 + SigLIP TensorRT engines
  -> coarse/fine GridPool and 31-frame history
  -> MiniCPM policy and trajectory head
  -> 8 [x, y, yaw] waypoints
```

`MiniCPMRobotTrackRequest` remains available for policy-only validation with
precomputed visual tokens. `MiniCPMRobotTrackImageRequest` is the normal image
inference interface.

## Preprocessing

Use `MiniCPMRobotTrackProcessor` to match the reference PIL resize and compact
token layout. The Engine accepts already processed `384x384` uint8 RGB frames;
it rejects other image sizes instead of silently using a different resize
implementation.

```python
processor = MiniCPMRobotTrackProcessor(
    tokenizer=tokenizer,
    history_frames=31,
    text_capacity=config.text_capacity,
    resize_workers=None,
)
inputs = processor.preprocess({"images": frames, "task": instruction})
```

The postprocessor moves the `(1, 8, 3)` waypoint tensor to CPU float32:

```python
waypoints = processor.postprocess(output.waypoints)
processor.close()
```

A complete 32-frame window resizes frames concurrently with the exact same PIL
BICUBIC operation. The default worker count is `min(8, CPU count)`; set
`resize_workers=1` for the serial reference path. Single-frame increments do
not enter the thread pool.

## Stream contract

Every request has a `stream_id`. A complete 32-frame request always replaces
that stream's history, even if the stream already exists. It never discards the
first 31 frames based on an unverified overlap assumption.

```python
cold = engine.step(
    MiniCPMRobotTrackImageRequest(
        frames=inputs.frames,
        input_ids=inputs.input_ids,
        text_lengths=inputs.text_lengths,
        stream_id="go2-front",
        frame_index=31,
    )
)
```

After the complete window, send one new frame with the next consecutive index:

```python
steady = engine.step(
    MiniCPMRobotTrackImageRequest(
        frames=next_frame,
        input_ids=inputs.input_ids,
        text_lengths=inputs.text_lengths,
        stream_id="go2-front",
        frame_index=32,
    )
)
```

Repeating frame index 32 is an idempotent retry: the committed visual features
are reused and the window does not advance. Skipped or older indices are
rejected. A single-frame request for an unknown or evicted stream is also
rejected; send a complete window to recreate it. The default cache retains up
to eight streams using LRU eviction and can be changed with
`max_cached_streams`.

Vision state is committed only after policy inference succeeds. A failed policy
call therefore leaves the previous stream state intact.

## TensorRT engines

Use the export and build scripts in the
[upstream MiniCPM-RobotTrack repository](https://github.com/OpenBMB/MiniCPM-Robot/tree/main/MiniCPM-RobotTrack):

```bash
bash scripts/export_onnx.sh
bash scripts/build_engines.sh
```

The plugin validates the static TensorRT contracts at startup:

| Engine | Input | Output |
| --- | --- | --- |
| DINOv3 | float32 `[1, 3, 384, 384]` | float32 `[1, 576, 384]` |
| pooled SigLIP | float32 `[1, 3, 384, 384]` | float32 `[1, 576, 1152]` |

The engines use FP16 internally but expose float32 bindings. TensorRT plan
files are tied to the TensorRT version and target GPU; build them on the target
Jetson software stack. Keep the ONNX export commit, TensorRT version, build
command, and engine SHA-256 with deployment artifacts.

## CUDA Graph behavior

The policy and steady single-frame vision paths use separate CUDA Graphs. The
vision graph captures normalization, both TensorRT enqueues, concatenation, and
pooling. A 32-frame replacement uses the eager loop because its shape differs.

The global `RuntimeConfig.use_cuda_graph` switch disables both graphs.
`use_vision_cuda_graph=False` can disable only the vision graph for A/B tests.

## Orin benchmark

```bash
python examples/minicpm_robot_track/benchmark_e2e.py \
  --checkpoint /path/to/MiniCPM-RobotTrack \
  --dino-engine /path/to/dino_patch_target_fp16.engine \
  --siglip-engine /path/to/siglip_pooled_target_maxn_fp16.engine \
  --frames-dir /path/to/ordered/rgb/frames \
  --warmup 5 \
  --iters 20 \
  --cold-iters 3 \
  --resize-workers 8
```

The script preprocesses the initial 32-frame window with the canonical
Processor, then benchmarks explicit single-frame increments and repeated full
window replacements. It reports both Engine-only CUDA timing and raw-RGB to
waypoint wall time including the Processor. `input_seq_length=256` is the fixed
policy token length; the vision towers still use `384x384` images.

This integration follows the model and checkpoint contract published by
[OpenBMB/MiniCPM-Robot](https://github.com/OpenBMB/MiniCPM-Robot). Model
weights and TensorRT plans are not distributed with PhyAI.
