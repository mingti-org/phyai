# LingBot V2 example

Set `CHECKPOINT_DIR` and `PROCESSOR_DIR` in `run_lingbot_v2.py` to your local
LingBot V2 checkpoint and Qwen3-VL Processor directories. From the repository
root, run:

```bash
uv run python examples/lingbot_v2/run_lingbot_v2.py
```

The example runs one observation with three synthetic images and a zero state.
It prints the Action shape and the first action. These inputs check the inference
path; the output is not a robot control command.

## Enable Wavefront

In `run_lingbot_v2.py`, add `use_wavefront_pipeline=True` to `LingBotV2Args`:

```python
plugin_args=LingBotV2Args(
    checkpoint_dir=CHECKPOINT_DIR,
    config=config,
    max_vision_tokens_per_image=max_vision_tokens,
    use_wavefront_pipeline=True,
),
```

In the same script, point `KernelConfig` to the Wavefront policy:

```python
kernel=KernelConfig(
    config_path=str(Path(__file__).with_name("kernel_policy_wavefront.yaml")),
),
```

Keep `use_cuda_graph=True` and run the same command. Wavefront schedules State
and Action layers on separate streams inside the Euler CUDA Graph. It is off by
default and requires a single GPU, SDPA paged attention, and Torch GEMM for the
concurrent expert linear layers.

The regular `kernel_policy.yaml` uses FlashInfer GEMM for the router. Wavefront
rejects that policy during initialization because its GEMM workspace can be
shared across streams. `kernel_policy_wavefront.yaml` selects Torch GEMM for
the router as well as the other linear layers.

Wavefront does not support vGPU or CUDA green-context streams. Keep it disabled
when using `vGPU.activate()` or `capture_on_stream(vgpu.stream)`. Ordinary CUDA
streams, including non-default streams, remain supported.
