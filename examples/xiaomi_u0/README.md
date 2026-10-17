# Xiaomi Robotics U0 text-to-image inference

This example generates a single 1024 × 1024 image from a text prompt through the registered `xiaomi_u0` PhyAI engine. Text preprocessing and image postprocessing live in `phyai-utils-tools`; AR inference uses the native PhyAI model, scheduler and runner.

## Supported configuration

- One CUDA device, batch size 1, BF16 AR parameters.
- Fixed 64 × 64 visual-token grid, no-text classifier-free guidance at scale 3.
- Explicit `torch_reference` paged attention and Torch LM Head; CUDA Graph is disabled.
- Released local Xiaomi-Robotics-U0-4B and Emu3.5-VisionTokenizer checkpoints.
- The released VisionTokenizer is loaded in FP32 using Transformers with `trust_remote_code=True`. Its checkpoint Python code is executed for image decode. Text tokenization reads vocabulary files directly.

FlashInfer generation quality, multiple GPUs, concurrent requests, other resolutions, video and interleaved tasks are not covered by this reference delivery. RLCR optimization is subsequent work.

## Dependencies and checkpoints

Use the repository's PhyAI runtime installation, including `phyai-kernel`, `phyai-utils-tools` and `phyai-ext`. After the runtime is installed, install the utility extra using the same Python environment:

```bash
python -m pip install -e './phyai-utils-tools[xiaomi_u0]'
```

This is a normal package installation and resolves repository-pinned dependencies. On a pre-provisioned, version-locked Thor environment, review the resolution before changing the runtime. The model-specific extra requires `tiktoken>=0.7,<1`.

The U0 checkpoint directory must contain `config.json`, released safetensors weights and index, `unis.tiktoken`, and `unis_vision_tokens.txt`. The VisionTokenizer directory must contain its Hugging Face configuration, released model implementation and weights. Downloads are outside this example: all model loading uses local files.

Validated runtime: NVIDIA Thor (aarch64, sm_110), Python 3.12, Torch 2.11.0+cu130, torchvision 0.26.0, Transformers 5.8.1 and FlashInfer 0.6.17. This is the tested environment, not a portability claim for all CUDA devices.

## Generate an image

From the repository root, in the installed runtime environment:

```bash
python examples/xiaomi_u0/run_xiaomi_u0.py \
  --checkpoint-dir /path/to/Xiaomi-Robotics-U0-4B \
  --vision-tokenizer-dir /path/to/Emu3.5-VisionTokenizer \
  --prompt "A clean product photograph of a glass teapot beside green tea leaves, soft daylight, neutral background." \
  --output-dir ./outputs/u0-teapot \
  --seed 42
```

Use a new or empty output directory. The example defaults to 5120 maximum generated tokens. Model initialization, generation including prefill, and VisionTokenizer loading plus decode are timed separately. The AR engine is closed before loading the VisionTokenizer.

To inspect tokenization without creating an AR engine or loading the VisionTokenizer:

```bash
python examples/xiaomi_u0/run_xiaomi_u0.py \
  --checkpoint-dir /path/to/Xiaomi-Robotics-U0-4B \
  --prompt "A compact industrial inspection camera on a brushed steel workbench, precise studio lighting." \
  --output-dir ./outputs/u0-prepare \
  --prepare-only
```

## Thor container invocation

The following assumes an existing validated virtual environment, compatible NVIDIA runtime, and already-downloaded checkpoints. Run from the repository root; replace the host environment and checkpoint paths as needed:

```bash
REPO="$PWD"
ENV_DIR=/path/to/u0_environment
AR_WEIGHTS=/path/to/Xiaomi-Robotics-U0-4B
VISION_WEIGHTS=/path/to/Emu3.5-VisionTokenizer
OUTPUT="$PWD/outputs/u0-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT"
set -o pipefail

docker run --rm --pull never --runtime=nvidia --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONNOUSERSITE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/repo/phyai/src:/repo/phyai-utils-tools/src \
  -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  -e TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \
  -v "$REPO:/repo:ro" -v "$ENV_DIR:/env:ro" \
  -v "$AR_WEIGHTS:/weights:ro" -v "$VISION_WEIGHTS:/vision:ro" \
  -v "$OUTPUT:/results" --entrypoint /env/venv/bin/python \
  nvcr.io/nvidia/pytorch:26.03-py3 \
  /repo/examples/xiaomi_u0/run_xiaomi_u0.py \
  --checkpoint-dir /weights --vision-tokenizer-dir /vision \
  --prompt "A clean product photograph of a glass teapot beside green tea leaves, soft daylight, neutral background." \
  --output-dir /results/output --seed 42 \
  2>&1 | tee "$OUTPUT/run.log"
```

## Output contract

The output directory contains `input_ids.npy`, `unconditional_ids.npy`, `tokenizer_mapping.json`, and `generation.json`. A completed generation also contains `generated_ids.npy` and `image.png`. Generated IDs exclude the prompt. Output files written before a failed generation or decode do not indicate completion; successful completion requires `image.png` and image-size metadata in `generation.json`.

The accepted single-image stream contains 7 header tokens, 4096 visual tokens, 63 EOL separators, EOI, ESS and EOS: 4169 generated tokens. No EOL follows the last visual row. The postprocessor rejects incomplete or malformed streams.

## Tests and validation

CPU processor tests:

```bash
python -m pytest phyai-utils-tools/tests/test_xiaomi_u0_processor.py \
  --import-mode=importlib -q
```

Real-vocabulary cases require an explicitly supplied checkpoint; otherwise those cases skip:

```bash
U0_TEST_CHECKPOINT=/path/to/Xiaomi-Robotics-U0-4B \
  python -m pytest phyai-utils-tools/tests/test_xiaomi_u0_processor.py \
  --import-mode=importlib -q
```

The full real-vocabulary run passed 17 tests on Thor. These cover preprocessing and sequence contracts, not the entire GPU model or runtime. Separate experimental regressions verified:

- Four unique prompt branches: 39/39 prefill nodes equal to the same-environment reference.
- Full fixed-history dual-branch replay: 4096 visual predictions with identical processed distributions.
- Native entry sampling and image decode for three prompts; same-engine sequential tea-pot requests reproducible.
- This public prompt-to-image example: 4169 sampled tokens identical to the accepted same-environment tea-pot result; actual EOS termination; 1024 × 1024 decode.

The public example recorded 9.12 seconds for engine setup, 930.55 seconds for generation including prefill, and 5.52 seconds for VisionTokenizer load plus decode. This is a single observed run, not a controlled performance benchmark or an optimization claim.

Old Torch 2.9 reference output and the Torch 2.11 runtime can differ numerically. Same-seed token equality across these environments is not promised. Decoding identical tea-pot tokens in the two environments produced a maximum 8-bit pixel difference of 1; quality evaluation remains separate from this numerical observation.

## Runtime contract regression

The validation suite passed 33 tests: 17 processor tests and 16
scheduler contract tests. The scheduler tests use simulated runners to
cover header mapping validation, target-device seeding, EOS and length
termination, request reset, exception recovery and invalid requests.
They do not replace GPU numerical validation.

After the mapping and seed-isolation fixes, the real GPU native-entry
regression passed 81 teacher-forced predictions, second-request reset
and replay EOS termination.

After the mapping and seed-isolation fixes, the final revised source
also passed free-sampling image generation: 4169 generated tokens,
actual EOS termination and 1024 x 1024 image decode. Generated tokens
were identical to the previously accepted same-environment result.

The final run recorded 7.58 seconds for engine setup, 950.17 seconds
for generation including prefill, and 5.41 seconds for VisionTokenizer
load plus decode. These are single-run observations, not benchmark results.
