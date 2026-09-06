<a name="english"></a>

**[English](#english) | [中文](#chinese)**

# breezetts2_mac_fast.py

A rewritten streaming generation loop for **Breeze-TTS-2** on Apple Silicon.

Playback stuttering when running Breeze-TTS-2 on a Mac is **not** caused by model size or quantization precision. Even with the ~3 GB 4bit weights, measured RTF stays around 1.85 — meaning 1 second of audio takes roughly 1.85 seconds to generate, so real-time playback will inevitably drop out. This project replaces the generation loop with one that removes the redundant computation responsible for most of that overhead, and ships with the diagnostics needed to tell a speed problem from an expression problem.

---

## Quick start

```bash
uv run --python 3.12 breezetts2_mac_fast.py
```

Defaults: 4bit weights, playback on, chunked decoding, RTF and underflow reporting.

If RTF is still ≥ 1, try whole-frame compiled mode:

```bash
uv run --python 3.12 breezetts2_mac_fast.py \
  --depth-mode compiled
```

The first run pays a compilation cost; the script reports warm-up time separately so it doesn't pollute your RTF number.

### With text and emotion control

```bash
uv run --python 3.12 breezetts2_mac_fast.py \
  --depth-mode compiled \
  --cfg-scale 1 \
  --chunk-frames 4 \
  --prebuffer-ms 640 \
  --text "看到你回来我真的很开心，可想到这些天一直等不到你的消息，心里又有点委屈。" \
  --instruct "同一个人连续自然地说话，保持连贯的语速和气息，情绪从开心渐渐流露出委屈，最后变得柔软安心。" \
  --runs 1 \
  --output breeze_emotion_smooth
```

### Options

| Flag | Default | Notes |
|---|---|---|
| `--model` | `mlx-community/Breeze-TTS-2-mlx-4bit` | Already 4bit; re-specifying it changes nothing |
| `--depth-mode` | `cached` | `cached` uses intra-frame KV reuse; `compiled` compiles the whole frame |
| `--chunk-frames` | `2` | Frames per decode chunk. Larger = fewer callbacks, higher first-sound latency |
| `--prebuffer-ms` | — | Audio buffered before playback starts. Only helps if RTF < 1 |
| `--cfg-scale` | `1` | Higher gives stronger instruction adherence at higher compute cost |
| `--no-play` | off | Generate to WAV only, no audio output |
| `--runs` | `2` | Second run excludes warm-up; read RTF from that one |

---

## What this fixes

### The real cause of the stutter

After checking the mlx-audio 0.5.1 source, the bottleneck is in how the Breeze **depth decoder** is implemented. Generating one frame of audio requires predicting several acoustic codes in sequence. In that version, each time the next code is predicted, the codes already produced within the current frame are fed back through the depth decoder from scratch — the KV cache for this stage is never reused. The result is a large volume of redundant computation plus extra GPU dispatch overhead, repeated for every code of every frame.

One clarification worth stating plainly: the low-latency figures Breeze publishes officially come from a warmed-up, optimized path on H100. That is not the same code path as the MLX implementation on a Mac, and the two numbers are not directly comparable.

### What was changed

- **Intra-frame KV cache reuse** — eliminates the redundant recomputation described above.
- **One CPU read-back per frame** — the per-code device sync is collapsed into a single transfer, cutting stall time.
- **Sensible defaults** — 4bit weights, playback enabled, chunked decoding out of the box.
- **Retained instrumentation** — RTF and underflow counts are still reported, so a slow run and a choppy-sounding run can be told apart.
- **`--depth-mode compiled`** — an alternative whole-frame compiled path; warm-up is timed and reported separately.

### Status

Validated against MLX numerical reference tests on small models, covering both 4bit and compiled modes. The actual speedup still needs to be measured on your own machine. Note that 3B parameters is not by itself a guarantee of real-time generation.

### Reading your results

- **RTF < 1** — generation outruns playback. If audio still drops out, raise `--prebuffer-ms`.
- **RTF ≥ 1** — buffering cannot fix this. Sustained underflow is expected; reduce `--cfg-scale`, try `--depth-mode compiled`, or generate offline to WAV.
- **The WAV itself sounds choppy** — this is an expression problem, not a speed problem. Dense exclamation marks, ellipses, and tags like `[笑]` / `[叹气]` ask the model to stop and switch register repeatedly. Use continuous prose with a gradual emotion instruction instead.

---
---

<a name="chinese"></a>

**[English](#english) | [中文](#chinese)**

# breezetts2_mac_fast.py（中文）

面向 Apple Silicon 重写的 **Breeze-TTS-2** 流式生成循环。

在 Mac 上运行 Breeze-TTS-2 时出现的播放卡顿，**并非**模型体积或量化精度所致——即便使用约 3 GB 的 4bit 权重，实测 RTF 仍在 1.85 左右，即生成 1 秒音频需要约 1.85 秒，播放必然缺音。本项目重写了生成循环，消除了造成大部分开销的重复计算，并保留了必要的诊断信息，用于区分「速度问题」与「表达问题」。

---

## 快速开始

```bash
uv run --python 3.12 breezetts2_mac_fast.py
```

默认配置：4bit 权重、开启播放、逐块解码、输出 RTF 与 underflow 统计。

如果 RTF 仍然 ≥ 1，可尝试整帧编译模式：

```bash
uv run --python 3.12 breezetts2_mac_fast.py \
  --depth-mode compiled
```

首次运行需要编译，耗时较长；脚本会单独显示预热时间，不会混入 RTF 数值。

### 带文本与情绪控制

```bash
uv run --python 3.12 breezetts2_mac_fast.py \
  --depth-mode compiled \
  --cfg-scale 1 \
  --chunk-frames 4 \
  --prebuffer-ms 640 \
  --text "看到你回来我真的很开心，可想到这些天一直等不到你的消息，心里又有点委屈。" \
  --instruct "同一个人连续自然地说话，保持连贯的语速和气息，情绪从开心渐渐流露出委屈，最后变得柔软安心。" \
  --runs 1 \
  --output breeze_emotion_smooth
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | `mlx-community/Breeze-TTS-2-mlx-4bit` | 默认已是 4bit，再次指定不会带来变化 |
| `--depth-mode` | `cached` | `cached` 为帧内 KV 复用；`compiled` 为整帧编译 |
| `--chunk-frames` | `2` | 每块解码帧数。数值越大回调越少，但首响延迟更高 |
| `--prebuffer-ms` | — | 播放前预缓冲时长。仅在 RTF < 1 时有效 |
| `--cfg-scale` | `1` | 数值越高指令控制越强，计算量也越大 |
| `--no-play` | 关闭 | 仅生成 WAV，不播放 |
| `--runs` | `2` | 第二轮不含预热，请以第二轮 RTF 为准 |

---

## 这一版解决了什么

### 卡顿的真实原因

经核对 mlx-audio 0.5.1 源码，瓶颈出在 Breeze **深度解码器**的实现方式上。每生成一帧声音，需要依次预测多个声学码；而该版本在预测下一个码时，会把当前帧内已有的码重新送入深度解码器从头计算一遍，完全没有复用这一阶段的 KV 缓存。由此产生大量重复运算，以及额外的 GPU 调用开销，并且在每一帧的每一个码上重复发生。

另有一点需要明确说明：Breeze 官方公布的低延迟数据来自 H100 上预热后的优化路径，与 Mac 上的 MLX 实现并非同一条链路，两者的数字不能直接对照。

### 具体改动

- **帧内 KV 缓存复用** —— 消除上述重复计算。
- **每帧只回读 CPU 一次** —— 将逐码同步合并为单次传输，减少等待。
- **合理的默认配置** —— 开箱即为 4bit 权重、开启播放、逐块解码。
- **保留诊断信息** —— 继续输出 RTF 与 underflow 统计，用于区分「生成慢」和「听起来碎」。
- **`--depth-mode compiled`** —— 另一条整帧编译路径，预热耗时单独计时并显示。

### 当前状态

已通过 MLX 小模型数值对照测试，覆盖 4bit 与编译两种模式。实际提速幅度仍需在你自己的机器上验证。需要说明的是，参数量为 3B 本身并不构成实时生成的保证。

### 如何看你的结果

- **RTF < 1** —— 生成快于播放。若仍有断音，提高 `--prebuffer-ms`。
- **RTF ≥ 1** —— 增加缓冲无法解决，持续 underflow 属于预期现象；可降低 `--cfg-scale`、改用 `--depth-mode compiled`，或改为离线生成 WAV。
- **WAV 本身就割裂** —— 这是表达问题，不是速度问题。密集的感叹号、省略号以及 `[笑]` / `[叹气]` 等标记，本身就在要求模型反复停顿、切换表达。改用连贯行文加渐变情绪指令即可。
