<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/logo/light.png">
    <img alt="PhyAI" src="docs/logo/light.png" width="360">
  </picture>
</p>

<p align="center">
  <a href="https://mingti-org.github.io/phyai-blog/"><img alt="Blog" src="https://img.shields.io/badge/blog-PhyAI-8B5CF6"></a>
  <a href="https://phyai.mintlify.app/"><img alt="Docs" src="https://img.shields.io/badge/docs-phyai-2563EB"></a>
  <a href="https://github.com/mingti-org/phyai"><img alt="GitHub" src="https://img.shields.io/badge/github-mingti--org%2Fphyai-181717?logo=github"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/mingti-org/phyai.svg"></a>
  <a href="https://github.com/mingti-org/phyai/issues"><img alt="open issues" src="https://img.shields.io/github/issues-raw/mingti-org/phyai"></a>
  <a href="https://mingti-org.github.io/phyai/simple/"><img alt="Nightly" src="https://img.shields.io/badge/nightly-packages-60A5FA"></a>
</p>

----

**PhyAI** (pronounced "phi") is a **latency-first serving engine for Physical AI**.
It is designed first for latency critical workloads, such as policy and action
models that run in interactive systems.

<p align="center">
  <img src="assets/phyai-demo.gif" alt="PhyAI and OpenPI deployment comparison" width="100%">
</p>

## News

- [2026/10] Support [PhyAI gateway](https://phyai.mintlify.app/deployment/server-gateway) for multi-host serving and LeRobot clients.
- [2026/09] 🚀 Support PI0.5 RL rollout for PPO. RLinf integration is in progress via [RLinf PR #1587](https://github.com/RLinf/RLinf/pull/1587).
- [2026/07] 🚀 Day 0 support for MiniCPM-Robotic [blog](https://mingti-org.github.io/phyai-blog/blogs/260719-day-0-minicpm-robotic/).
- [2026/07] 👏 Introducing PhyAI, a latency-first serving engine for Physical AI. [Read the Blog](https://mingti-org.github.io/phyai-blog/blogs/260718-phyai/).
- [2026/07] Support [PI0](https://phyai.mintlify.app/models/pi0/ws1).
- [2026/07] Support Cosmos3-Super with TP and CFG parallelism in the unified Cosmos3 [generation path](https://phyai.mintlify.app/models/cosmos/generation).
- [2026/06] Support [Pi0.5](https://phyai.mintlify.app/models/pi05/ws1) and Cosmos3-Nano's [policy mode](https://phyai.mintlify.app/models/cosmos/policy) and [generation mode](https://phyai.mintlify.app/models/cosmos/generation).


## Key Features

- 🚀 Runs on NVIDIA Jetson edge devices
- 🚀 Scales to GPU clusters with DP, TP, and CFG parallelism
- 🚀 Uses high-performance kernels from FlashInfer and Humming
- 🤗 Supports W4A8 (NVFP4, MXFP4, INT4), W8A8, and W8A16 quantization (PR under review)

## Supported Models

<table align="center" width="100%">
  <tbody>
    <tr>
      <th align="center" width="22%">VLA</th>
      <td align="center">
        <a href="https://phyai.mintlify.app/models/pi0/ws1"><strong>&pi;0</strong></a>,
        <a href="https://phyai.mintlify.app/models/pi05/ws1"><strong>&pi;0.5</strong></a>(w/ DP),
        <a href="https://phyai.mintlify.app/models/gr00t/ws1"><strong>GR00T N1.7</strong></a>,
        <a href="examples/minicpm_gr00t/README.md"><strong>MiniCPM-Robot</strong></a>
      </td>
    </tr>
    <tr>
      <th align="center" width="22%">WAM</th>
      <td align="center">
        <a href="https://phyai.mintlify.app/models/cosmos/policy"><strong>Cosmos3-Nano-Policy-DROID</strong></a>(w/ TP, CFG Parallel)
      </td>
    </tr>
    <tr>
      <th align="center" width="22%">Foundation Model</th>
      <td align="center">
        <a href="https://phyai.mintlify.app/models/cosmos/generation"><strong>Cosmos3-Nano</strong></a>(w/ TP, CFG Parallel),
        <a href="https://phyai.mintlify.app/models/cosmos/generation"><strong>Cosmos3-Super</strong></a>(w/ TP, CFG Parallel),
        <a href="phyai/src/phyai/models/qwen3_5"><strong>Qwen3.5</strong></a>,
        <a href="phyai/src/phyai/models/qwen3_vl"><strong>Qwen3-VL</strong></a>
      </td>
    </tr>
  </tbody>
</table>

## Performance Comparison

<p align="center">
  <img src="assets/performance-comparison.svg" alt="Bar chart comparing PhyAI and official single-request latency across supported models and devices" width="100%">
</p>

## Installation

See the [PhyAI installation guide](https://phyai.mintlify.app/) for the latest
source and nightly package instructions.

**From source:**

```bash
git clone https://github.com/mingti-org/phyai
cd phyai
uv sync
```

**Nightly build:**

```bash
uv pip install phyai phyai-ext \
  --extra-index-url https://mingti-org.github.io/phyai/simple/ \
  --prerelease=allow
```

## Multi-host serving

You can run model servers on separate machines and connect them through
`phyai-gateway`. On each model host, prepare `pi05.yaml` with that host's
checkpoint, tokenizer, and robot settings. You can start from the
[PI0.5 example](examples/pi05/server.yaml), which serves `pi05` on port `50063`.
Check the configuration, then start the server:

```bash
phyai server pi05.yaml
```

On the gateway host, register both servers at startup. Replace the IP addresses
below with your server hosts; `pi05` must match `server.model_name` in their YAML
files. Repeat `--backend MODEL HOST:PORT` to add more servers:

```bash
phyai-gateway \
  --backend pi05 10.0.0.11:50063 \
  --backend pi05 10.0.0.12:50063
```

In another terminal on the gateway host, check that the backends report
`healthy: true`:

```bash
curl http://127.0.0.1:30000/v1/backends
```

Clients use the gateway host's HTTP port `30000` or gRPC port `50111`. The gateway
distributes stateless requests across healthy replicas. See the
[server and gateway setup guide](https://phyai.mintlify.app/deployment/server-gateway) for client
configuration and adding or removing servers while the gateway is running.

## Contribution Guidelines

We thank the contributors below and welcome more developers to join us in building PhyAI.

<a href="https://github.com/mingti-org/phyai/graphs/contributors"><img src="https://stg.contrib.rocks/image?repo=mingti-org/phyai&max=240&columns=18" /></a>

## Sponsors & Adoption

PhyAI is a latency-first, open-source serving engine for Physical AI. It is being adopted by companies working across AI infrastructure and robotics, including Mingti and ModelBest.

We are actively seeking partnerships with compute providers, chip vendors, and robotics companies. If you are interested in working with us, please [contact us](#contact).

<table align="center" width="80%">
  <tbody>
    <tr>
      <td align="center" width="50%">
        <img src="assets/mingti-new.png" alt="Mingti" width="240">
      </td>
      <td align="center" width="50%">
        <img src="assets/modelbest.svg" alt="ModelBest" width="240">
      </td>
    </tr>
  </tbody>
</table>

## Citation

If you use PhyAI in research or production work, please cite the project:

```bibtex
@software{phyai2026,
  title = {PhyAI: Latency-First Serving Engine for Physical AI},
  author = {{PhyAI Team}},
  year = {2026},
  url = {https://github.com/mingti-org/phyai}
}
```

<a id="contact"></a>
**Contact:**

We welcome PhD and master's students who want to help build Physical AGI, especially those interested in systems infrastructure. We also want to work with chip and compute companies, as well as robotics companies that plan to deploy models with PhyAI.

- PhyAI: [Maintainer](mailto:chenghua.wang.edu@gmail.com)
- Mengwei Xu: [mwx@bupt.edu.cn](mailto:mwx@bupt.edu.cn)
- Daliang Xu: [xudaliang@bupt.edu.cn](mailto:xudaliang@bupt.edu.cn)

## License

PhyAI is released under the [MIT License](LICENSE). It uses [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Humming](https://github.com/inclusionAI/humming), and [FLA](https://github.com/fla-org/flash-linear-attention). We have also learned a great deal from [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), and [TokenSpeed](https://github.com/lightseekorg/tokenspeed). We thank the maintainers and contributors of all these projects.

## Demos

### Cosmos3-Nano-Policy-DROID, 260718 nightly version

https://github.com/user-attachments/assets/12a833ce-3b47-4f08-875c-b30cc2567bef

https://github.com/user-attachments/assets/8e7d433c-e3b1-4388-8488-196836a1ba51

https://github.com/user-attachments/assets/29067c88-f51a-4ca6-ab8c-b45f4d778923
