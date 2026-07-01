<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/logo/light.png">
    <img alt="PhyAI" src="docs/logo/light.png" width="360">
  </picture>
</p>

<p align="center">
  <a href="https://phyai.mintlify.app/"><img alt="Docs" src="https://img.shields.io/badge/docs-phyai-2563EB"></a>
  <a href="https://github.com/MEmbodied/phyai"><img alt="GitHub" src="https://img.shields.io/badge/github-MEmbodied%2Fphyai-181717?logo=github"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/MEmbodied/phyai.svg"></a>
  <a href="https://github.com/MEmbodied/phyai/issues"><img alt="open issues" src="https://img.shields.io/github/issues-raw/MEmbodied/phyai"></a>
  <a href="https://membodied.github.io/phyai/simple/"><img alt="Nightly" src="https://img.shields.io/badge/nightly-packages-60A5FA"></a>
</p>

----

**PhyAI** (pronounced "phi") is a **latency-first serving engine for Physical AI**.
It is designed first for latency critical workloads, such as policy and action
models that run in interactive systems.

## News

- [2026/07] 🚀 Support Cosmos3-Super (TP + CFG parallel) in the Cosmos3 [WN generation path](https://phyai.mintlify.app/models/cosmos/wn).
- [2026/06] 🚀 Support [Pi0.5](https://phyai.mintlify.app/models/pi05/ws1) and Cosmos3-Nano's [policy mode](https://phyai.mintlify.app/models/cosmos/ws1_policy) & [gen mode](https://phyai.mintlify.app/models/cosmos/ws1).

## Installation

See the [PhyAI installation guide](https://phyai.mintlify.app/) for the latest
source and nightly package instructions.

**From source:**

```bash
git clone https://github.com/MEmbodied/phyai
cd phyai
uv sync
```

**Nightly build:**

```bash
uv pip install phyai phyai-ext \
  --extra-index-url https://membodied.github.io/phyai/simple/ \
  --prerelease=allow
```

## Citation

If you use PhyAI in research or production work, please cite the project:

```bibtex
@software{phyai2026,
  title = {PhyAI: Latency-First Serving Engine for Physical AI},
  author = {{PhyAI Team}},
  year = {2026},
  url = {https://github.com/MEmbodied/phyai}
}
```

## License

PhyAI is released under the [MIT License](LICENSE).

## Notice

PhyAI is under active development. APIs, package layout, and deployment paths may
change before stable releases.

PhyAI is grateful for excellent open-source implementations from the community,
including [SGLang](https://github.com/sgl-project/sglang) and
[LeRobot](https://github.com/huggingface/lerobot). When using, modifying, or
redistributing PhyAI, keep the relevant attribution and comply with applicable
upstream license and notice requirements.
