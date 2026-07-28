# MiniCPM-RobotTrack本地部署

本指南使用PhyAI Engine运行MiniCPM-RobotTrack的完整推理链路。
输入是32帧RGB图像和一条文本指令，输出是8个`[x, y, yaw]`
waypoint。示例只包含本地推理，不启动网络服务。

```text
32帧RGB + 文本指令
  -> PIL BICUBIC resize + tokenizer
  -> DINOv3 + SigLIP
  -> 31帧coarse token历史 + 当前帧fine token
  -> MiniCPM4 policy + trajectory head
  -> 8个[x, y, yaw] waypoint
```

## 环境要求

- Linux和支持CUDA的NVIDIA GPU
- Python 3.12或更高版本
- [`uv`](https://docs.astral.sh/uv/)
- MiniCPM-RobotTrack checkpoint
- 本地DINOv3和SigLIP checkpoint
- 按时间顺序命名的RGB图像目录

PhyAI不分发模型权重。接入遵循
[OpenBMB/MiniCPM-Robot](https://github.com/OpenBMB/MiniCPM-Robot/tree/main/MiniCPM-RobotTrack)
发布的checkpoint和模型结构。

## 安装

在仓库根目录执行：

```bash
uv sync
```

`uv sync`会创建`.venv`并安装workspace中的包。下面的命令都使用
`uv run`，避免误用系统Python。

### Jetson Orin与CUDA12.6

仓库主workspace仍要求Python 3.12。Jetson AI Lab目前为JetPack 6的
CUDA12.6索引提供了CPython 3.10版PyTorch 2.11 wheel，其中包含Orin需要的
`sm_87`设备代码。Orin可使用示例目录中的独立project，不需要修改根
`pyproject.toml`：

```bash
unset VIRTUAL_ENV
uv sync \
  --project examples/minicpm_robot_track/jetson-cu126 \
  --python /usr/bin/python3.10

uv build --package phyai-ext --wheel \
  --out-dir .cache/jetson-cu126-wheels
uv pip install --python 3.12 \
  --target .cache/jetson-cu126-runtime \
  --no-deps --reinstall \
  .cache/jetson-cu126-wheels/phyai_ext-*.whl

export PYTHONPATH="$PWD/.cache/jetson-cu126-runtime:$PWD/phyai/src:$PWD/phyai-kernel:$PWD/phyai-utils-tools/src"
```

`phyai-ext`的wheel标记为`py3-none-linux_aarch64`，但包元数据沿用主
workspace的Python 3.12要求，因此上面的安装步骤使用Python 3.12将它展开到
独立目录。RobotTrack进程仍使用独立project中的Python 3.10：

```bash
examples/minicpm_robot_track/jetson-cu126/.venv/bin/python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())'
```

预期输出包含Torch `2.11.0`、CUDA `12.6`和`sm_87`。这套project只用于
Jetson运行时验证，不改变PhyAI对外声明的Python版本。

## 准备输入

`benchmark_e2e.py`会按文件名中的数字顺序读取图像，取最后32帧。
图像不足32帧时，脚本使用窗口中最早的一帧在左侧补齐。
Processor使用PIL BICUBIC将每帧resize到`384x384`。

Engine支持两种图像请求：

| 请求 | 输入 | 用途 |
| --- | --- | --- |
| 完整窗口 | 32帧RGB | 创建或替换stream的视觉历史 |
| 增量请求 | 1帧RGB | 为已初始化的stream推进一帧 |

## 运行一次推理

视觉塔直接加载DINOv3和SigLIP checkpoint，并通过PhyAI执行。

```bash
uv run python examples/minicpm_robot_track/benchmark_e2e.py \
  --checkpoint /path/to/MiniCPM-RobotTrack \
  --dino-checkpoint /path/to/dinov3-vit-s16 \
  --siglip-checkpoint /path/to/siglip-so400m-patch14-384 \
  --vision-attention-backend sdpa \
  --vision-norm-backend phyai-kernel \
  --vision-dtype float16 \
  --frames-dir /path/to/ordered/rgb/frames \
  --instruction "Follow the person in the red shirt." \
  --warmup 0 \
  --iters 1 \
  --cold-iters 1
```

使用上面的Jetson独立project时，把命令开头的`uv run python`替换为：

```text
examples/minicpm_robot_track/jetson-cu126/.venv/bin/python
```

脚本先提交32帧初始化窗口，再提交一个单帧增量请求。成功运行后，
JSON输出中的`waypoints_shape`应为`[1, 8, 3]`，waypoint为有限的
float32数值。

## 滑动窗口

每个请求都需要`stream_id`。新stream必须先提交32帧完整窗口，
`frame_index`是窗口最后一帧的序号。后续单帧请求使用连续序号。

| 情况 | Engine行为 |
| --- | --- |
| 新`stream_id`提交32帧 | 初始化stream |
| 已存在的`stream_id`提交32帧 | 替换完整历史 |
| 提交下一个`frame_index` | 只编码新帧并推进窗口 |
| 重复最新的`frame_index` | 复用已提交的视觉特征 |
| 跳过序号或使用更旧的序号 | 报错 |
| 未知或已淘汰的stream提交单帧 | 报错，需要重新提交32帧 |

视觉状态只在policy推理成功后提交。默认最多缓存8个stream，
超过上限后按LRU淘汰。

## Benchmark

稳态benchmark会先建立31帧历史，再反复追加一帧并输出8个
waypoint。使用10次预热和50次计时：

```bash
uv run python examples/minicpm_robot_track/benchmark_e2e.py \
  --checkpoint /path/to/MiniCPM-RobotTrack \
  --dino-checkpoint /path/to/dinov3-vit-s16 \
  --siglip-checkpoint /path/to/siglip-so400m-patch14-384 \
  --frames-dir /path/to/ordered/rgb/frames \
  --warmup 10 \
  --iters 50 \
  --cold-iters 1 \
  --json-output /tmp/robottrack_phyai.json
```

本机Orin的结果如下。测试使用PhyAI policy、PhyAI原生视觉、相同的
Processor和图像，并执行10次预热和50次计时。

| 指标 | P50时延 |
| --- | ---: |
| Vision Graph | 94.98 ms |
| Policy Graph | 30.34 ms |
| Engine | 126.40 ms |
| Processor | 6.82 ms |
| 原始RGB到waypoint | 133.92 ms |

`Engine P50`从Processor完成resize和tokenizer之后开始，不包含RPC和
网络时间。数值来自一套Orin测试环境，部署时应在目标设备重新测量。

### 只测policy

已经有coarse/fine视觉token时，使用`benchmark.py`排除Processor和视觉塔：

```bash
uv run python examples/minicpm_robot_track/benchmark.py \
  --checkpoint /path/to/MiniCPM-RobotTrack \
  --attention-backend flashinfer \
  --linear-kernel auto \
  --warmup 5 \
  --iters 30
```

## PhyAI视觉执行

PhyAI直接加载DINOv3和SigLIP checkpoint，以FP16执行视觉塔。
Attention使用PyTorch SDPA，LayerNorm使用`phyai-kernel`。接入不改变
DINOv3/SigLIP的层数、维度、token网格和输出，只替换权重装载方式、
算子实现和执行调度。

Policy和稳态单帧视觉分别使用一张CUDA Graph。32帧初始化因为
shape不同，使用eager执行。`--no-vision-cuda-graph`只关闭Vision
Graph，`--no-cuda-graph`同时关闭Vision Graph和Policy Graph。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  phyai-utils-tools/tests/test_minicpm_robot_track_processor.py

uv run ruff check \
  examples/minicpm_robot_track \
  phyai-utils-tools/src/phyai_utils_tools/models/minicpm_robot_track \
  phyai/src/phyai/models/minicpm_robot_track
```

固定输入的数值对比只用来检查接入正确性。真实跟随效果仍需要在
RobotTrack评测或机器人闭环中验证。

## 当前限制

- 只支持单GPU和batch size 1。
- Policy输入长度固定为256 tokens，视觉输入固定为`384x384`。
- stream必须先提交32帧完整窗口，之后才能提交单帧增量请求。
- 示例不测量RPC、网络、相机帧龄和机器人控制时延。
