# Xiaomi Robotics U0 text-to-image inference

This example generates a single 1024 × 1024 image from a text prompt through the registered `xiaomi_u0` PhyAI engine. Text preprocessing and image postprocessing live in `phyai-utils-tools`; AR inference uses the native PhyAI model, scheduler and runner.

## Supported configuration

- One CUDA device, batch size 1, BF16 AR parameters.
- Fixed 64 × 64 visual-token grid, no-text classifier-free guidance at scale 3.
- Explicit `torch_reference` paged attention and Torch LM Head; CUDA Graph is disabled.
- Released local Xiaomi-Robotics-U0-4B and Emu3.5-VisionTokenizer checkpoints.
- The released VisionTokenizer is loaded in FP32 using Transformers with `trust_remote_code=True`. Its checkpoint Python code is executed for image decode. Text tokenization reads vocabulary files directly.

FlashInfer generation quality, multiple GPUs, concurrent requests, other resolutions, video and interleaved tasks are outside the validated scope. RLCR optimization is subsequent work.

## Dependencies and checkpoints

From the repository root, use `uv` to install the workspace, including `phyai-kernel`, `phyai-utils-tools` and `phyai-ext`, with the Xiaomi U0 extra:

```bash
uv lock
uv sync --locked --all-packages --extra xiaomi_u0
```

`uv lock` creates or updates the shared `uv.lock`; review any changes before syncing. `--all-packages` applies `--extra xiaomi_u0` to workspace members, including `phyai-utils-tools`, which declares this extra. The model-specific extra requires `tiktoken>=0.7,<1`. The sync and run commands use `--locked` to reject a missing or outdated lockfile instead of updating it implicitly. On a provisioned Thor environment, review the sync plan with `uv sync --locked --all-packages --extra xiaomi_u0 --dry-run` before changing the runtime.

The U0 checkpoint directory must contain `config.json`, released safetensors weights and index, `unis.tiktoken`, and `unis_vision_tokens.txt`. The VisionTokenizer directory must contain its Hugging Face configuration, released model implementation and weights. Downloads are outside this example: all model loading uses local files.

Validated runtime: NVIDIA Thor (aarch64, sm_110), Python 3.12.3, Torch 2.11.0 with CUDA 13.0, torchvision 0.26.0, Transformers 5.8.1, FlashInfer 0.6.17 and tiktoken 0.14.0. These are the versions recorded in the latest regression; other CUDA devices and environments require separate validation.

## Generate an image

From the repository root, using the workspace environment prepared above:

```bash
uv run --locked --all-packages --extra xiaomi_u0 python examples/xiaomi_u0/run_xiaomi_u0.py \
  --checkpoint-dir /path/to/Xiaomi-Robotics-U0-4B \
  --vision-tokenizer-dir /path/to/Emu3.5-VisionTokenizer \
  --prompt "A clean product photograph of a glass teapot beside green tea leaves, soft daylight, neutral background." \
  --output-dir ./outputs/u0-teapot \
  --seed 42
```

Use a new or empty output directory. The example defaults to 5120 maximum generated tokens. Model initialization, generation including prefill, and VisionTokenizer loading plus decode are timed separately. The AR engine is closed before loading the VisionTokenizer.

To inspect tokenization without creating an AR engine or loading the VisionTokenizer:

```bash
uv run --locked --all-packages --extra xiaomi_u0 python examples/xiaomi_u0/run_xiaomi_u0.py \
  --checkpoint-dir /path/to/Xiaomi-Robotics-U0-4B \
  --prompt "A compact industrial inspection camera on a brushed steel workbench, precise studio lighting." \
  --output-dir ./outputs/u0-prepare \
  --prepare-only
```

## Thor container invocation

The following assumes an existing validated virtual environment, compatible NVIDIA runtime, and downloaded checkpoints. Run from the repository root; replace the host environment and checkpoint paths as needed. Check GPU usage before starting generation.

```bash
REPO="$PWD"
ENV_DIR=/path/to/u0_environment
AR_WEIGHTS=/path/to/Xiaomi-Robotics-U0-4B
VISION_WEIGHTS=/path/to/Emu3.5-VisionTokenizer
OUTPUT="$PWD/outputs/u0-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT"
set -o pipefail
nvidia-smi

docker run --rm --pull never --runtime=nvidia --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e PYTHONNOUSERSITE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 -e XDG_CACHE_HOME=/tmp/cache -e TRITON_CACHE_DIR=/tmp/triton \
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

Model-level validation is retained separately from the repository's regular test suite. Reports are saved under `.cache/u0/`. The external validation bundle is required to reproduce the numerical regression; its scripts are not included in this example or installed with the package.

### Commit-bound regression

The following results were obtained on Thor for commit `ea99d6f77045b7a8d689649f3a9832c378393280`. The downloaded raw reports, source and verification-code manifests, token arrays and images were reviewed locally on 2026-10-21. That review did not rerun GPU inference.

| Check | Result |
| --- | --- |
| Verifier CPU acceptance checks | 13 passed in both numerical regression runs |
| Short fixed-history replay | 81 predictions; 6279 exact activation-node comparisons |
| Complete tea-pot fixed-history replay | 4169 predictions; 325104 exact activation-node comparisons; 4096 visual predictions |
| Checkpoint parameters | All 399 candidate parameter tensors reported equal to independently loaded checkpoint tensors |
| CFG and candidate support | Processed scores exactly equal at every recorded step; maximum probability total variation 0; support differences 0 |
| KV and image boundaries | 63 EOL predictions and 14 selected KV comparison boundaries checked |
| Free sampling and image decoding | Three prompts at seed 42; complete 4169-token streams, sampled EOS termination and 1024 × 1024 images |
| Sequential requests | Tea-pot, camera, gripper, then tea-pot on one EngineCore; repeated tea-pot tokens and image pixels exactly equal |

The numerical verifier exercises the real EngineCore and the PR's `torch_reference` backend. It compares against archived independent PyTorch reference functions and the archived official logits processor. Observation hooks return candidate outputs unchanged. The archived experimental Runner and backend are not used.

The full replay requires a passed short report with matching commit, checkpoint hash records, inputs, runtime, container image and verification-code hashes. All 4169 full-replay JSONL rows were checked, and the first 81 rows match the short replay. The normal full-replay cache lengths are 4208 conditional tokens and 4188 unconditional tokens. The final EOS is returned without another model forward.

The local evidence review passed 159 archive-consistency checks. The retained archive is `pr71-final-evidence-20261021.tar.gz`, with SHA256:

```text
2e135560a80432f9d93f22d8f8c239812784a2fce500bab5bf30e7d2429ed1da
```

### Validation scope

Numerical parity covers one recorded tea-pot history in the supported configuration. Free-sampling regression covers three prompts at one seed. These results do not establish statistical image quality or correctness for every prompt.

Intermediate activation and parameter equality is reported by the checksum-matched verifier. The evidence archive does not retain every compared intermediate tensor. Checkpoint files were not downloaded for independent local rehashing. The sequential run records config and vocabulary hashes but not AR checkpoint shard or VisionTokenizer weight hashes; its source/runtime manifests and first tea-pot input/output hashes agree with numerical replay.

### Historical contract tests

Earlier delivery records report 33 passing contract tests: 17 processor cases and 16 simulated scheduler cases. The scheduler cases cover header mapping validation, target-device seeding, EOS and length termination, request reset, exception recovery and invalid requests.

Those 33 tests were not rerun in the regression above. Their files are not part of this PR's tracked test tree. They remain historical evidence and are separate from the 13 verifier CPU acceptance checks and the GPU numerical comparison. No repository-local pytest command for those archived tests is provided here.

Earlier records also report four prompt branches with 39/39 prefill nodes equal to a same-environment reference, and a complete run of the public prompt-to-image example. A historical public-example run recorded 7.58 seconds for engine setup, 950.17 seconds for generation including prefill, and 5.41 seconds for VisionTokenizer load plus decode. These are single-run observations, not a controlled benchmark or a performance improvement claim.

Torch 2.9 and Torch 2.11 environments can produce different numerical results. Same-seed token equality across those environments is not promised. A historical comparison of identical tea-pot tokens decoded in both environments found a maximum 8-bit pixel difference of 1; that observation is separate from image quality evaluation.