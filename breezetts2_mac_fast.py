# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlx-audio==0.5.1", "mlx==0.32.2", "sounddevice==0.5.3"]
# ///
"""Breeze TTS 2 Mac: cached depth decoding + one host read per acoustic frame.

直接运行（默认 4bit，默认有声音，Apple Silicon / 原生 arm64 Python）：
  uv run --python 3.12 breezetts2_mac_fast.py
  uv run --python 3.12 breezetts2_mac_fast.py --text "你好，我正在本地生成流式语音。"

可选的整帧图编译（首次编译可能较慢，预热耗时单独显示）：
  uv run --python 3.12 breezetts2_mac_fast.py --depth-mode compiled

相同文本、seed、分块参数下比较原生实现：
  uv run --python 3.12 breezetts2_mac_fast.py --depth-mode native --output breeze_native

检查实际加载权重的缓存 logits（额外耗时，之后正常播放）：
  uv run --python 3.12 breezetts2_mac_fast.py --verify-depth

首次成功运行后可离线：
  HF_HUB_OFFLINE=1 uv run --offline --python 3.12 breezetts2_mac_fast.py

变化：每个声学帧内缓存 depth 的 K/V，前两位置只预填充一次；后续每次只算
一个新位置。所有 depth 采样留在 MLX 数组上，一帧生成完成后才一次读回 CPU。
帧与帧之间不共享 depth 缓存。保留原生主干 KV 缓存、EOS、采样规则与流式 codec。
可选 compiled 模式明确捕获并更新 MLX RNG 状态。CFG 默认 1；CFG>1 时两分支
在 depth 阶段合并为 batch=2。不会删 codebook、加速播放或将文字切句。

默认 cached 模式不依赖整帧编译。--warmup-frames 默认 2，仅预热 depth；
预热单独计时且不计入 TTFA/RTF。第一次完整提示处理和音频解码仍计入 TTFA。
--chunk-frames 默认 2，首块与后续相同。--prebuffer-ms 默认 320。
--no-play 才会关闭播放，仍保存 WAV/JSON。--profile 会增加 GPU 同步开销。
输出 breeze_fast_1.wav/.json 和 breeze_fast_2.wav/.json；同前缀重跑会覆盖。
4bit 权重约 3 GB，运行内存还包括其他组件与缓存。仅支持一次一个请求。

本文件基于用户 Fish 播放/测速脚本及 Breeze 原生接口改写。优化算法已用实际
MLX CPU 运算及小型随机权重对照测试；未在真实 Mac 或完整 Breeze 权重上实测。
数值执行路径变化可能改变随机采样结果，不保证同 seed 的 WAV 逐样本相同。
3B 参数量不保证所有 Mac 都能实时；本脚本报告实际 RTF 和播放欠载。

播放修订：默认 spawn 独立进程 + RawOutputStream 阻塞写入，避开推理进程的
Python 回调调度。进程间按块传输，推理不等待声音播完。播放与 WAV 使用相同
限幅后的 PCM16；默认预缓冲 320ms、设备延迟请求 100ms，会增加播放首响延迟。
--player callback 保留旧浮点回调路径供对照。独立进程不伪造精确 DAC 或缺音
时长指标；打印实际设备延迟、音频提交时间和设备 underflow 次数。

单独检查同一播放链路（不加载模型）：
  uv run --python 3.12 breezetts2_mac_fast.py --play-wav breeze_fast_2.wav
  uv run --python 3.12 breezetts2_mac_fast.py --list-audio-devices
  uv run --python 3.12 breezetts2_mac_fast.py --device 设备编号


Sources:
https://github.com/Blaizzy/mlx-audio/blob/v0.5.1/mlx_audio/tts/models/breeze_tts/breeze_tts.py
https://github.com/Blaizzy/mlx-audio/blob/v0.5.1/mlx_audio/lm/models/llama.py
https://ml-explore.github.io/mlx/build/html/usage/compile.html
https://python-sounddevice.readthedocs.io/en/0.5.3/api/raw-streams.html
https://huggingface.co/mlx-community/Breeze-TTS-2-mlx-4bit
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import queue
import subprocess
import threading
import time
import wave
from importlib.metadata import version
from pathlib import Path


def hardware_info():
    info = {"chip": platform.processor() or platform.machine(), "memory_gib": None}
    if platform.system() == "Darwin":
        def sysctl(key):
            return subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", key], text=True, timeout=3).strip()
        try:
            info["chip"] = sysctl("machdep.cpu.brand_string")
            info["memory_gib"] = int(sysctl("hw.memsize")) / 2**30
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return info


class StageProfiler:
    """Opt-in synchronous timing, scoped to one model in this serial script.

    Temporarily wrap class methods because Python special-method dispatch ignores
    an instance __call__ attribute. Do not insert functions into MLX module state.
    Only the target instance is timed; restore methods on success and failure.
    """
    def __init__(self, model, enabled):
        self.model, self.enabled = model, enabled
        self.stats, self.patches = {}, []

    def _wrap(self, target, name, stage):
        import mlx.core as mx
        cls = type(target)
        original = getattr(cls, name)
        owned = name in cls.__dict__

        def measured(obj, *args, **kwargs):
            if obj is not target:
                return original(obj, *args, **kwargs)
            label = ("backbone_prefill" if kwargs.get("input_embeddings") is not None
                     else "backbone_decode") if stage == "backbone" else stage
            mx.synchronize()
            start = time.perf_counter()
            result = original(obj, *args, **kwargs)
            # The pinned methods return an MLX array or already-materialized
            # Python code IDs (_depth_tokens). Force lazy array results here.
            if isinstance(result, mx.array):
                mx.eval(result)
            mx.synchronize()
            entry = self.stats.setdefault(label, {"seconds": 0.0, "calls": 0})
            entry["seconds"] += time.perf_counter() - start
            entry["calls"] += 1
            return result

        self.patches.append((cls, name, original, owned))
        setattr(cls, name, measured)

    def install(self):
        if not self.enabled:
            return
        try:
            self._wrap(self.model, "_prompt_embeddings", "prompt")
            self._wrap(self.model, "_depth_tokens", "depth")
            self._wrap(self.model.backbone_model, "__call__", "backbone")
            self._wrap(self.model.audio_tokenizer.decoder, "streaming_step", "codec")
        except BaseException:
            self.close()
            raise

    def close(self):
        for cls, name, original, owned in reversed(self.patches):
            if owned:
                setattr(cls, name, original)
            else:
                delattr(cls, name)
        self.patches.clear()


class FrameKVCache:
    """Small append-only cache, newly allocated for EACH acoustic frame.

    At most num_codebooks entries; no 256-token capacity allocation is needed.
    Used with the pinned MLX-Audio Llama attention's offset/update_and_fetch API.
    """
    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        import mlx.core as mx
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate((self.keys, keys), axis=2)
            self.values = mx.concatenate((self.values, values), axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values


class FastDepth:
    """Cache depth attention and defer host reads to the end of each frame.

    This leaves the text encoder, backbone generation, EOS, and streaming audio
    codec in the release implementation. Sampling uses the release's sampler.
    Optional compilation captures random state explicitly. Cached is the default:
    compiled full-frame graphs may take time to compile on a particular Mac.
    """
    def __init__(self, model, mode="cached"):
        self.model, self.mode = model, mode
        self.functions = {}
        self.original = None

    def initial(self, first, hidden):
        import mlx.core as mx
        m = self.model.depth_decoder.model
        if m.backbone_hidden_state_projector is not None:
            hidden = m.backbone_hidden_state_projector(hidden)
        ids = mx.broadcast_to(first.reshape(1, 1), (hidden.shape[0], 1))
        embeds = mx.concatenate((hidden[:, None, :], m.embed_tokens(ids)), axis=1)
        x = m.inputs_embeds_projector(embeds)
        caches = [FrameKVCache() for _ in m.layers]
        for layer, cache in zip(m.layers, caches):
            x = layer(x, "causal", cache)
        return x, caches

    def advance(self, token, codebook_index, caches, batch):
        import mlx.core as mx
        m = self.model.depth_decoder.model
        ids = mx.broadcast_to(token.reshape(1, 1), (batch, 1))
        # At absolute depth position p>=1, upstream uses offset (p-1)*vocab.
        embeds = m.embed_tokens(ids + codebook_index * m.vocab_size)
        x = m.inputs_embeds_projector(embeds)
        for layer, cache in zip(m.layers, caches):
            x = layer(x, None, cache)
        return x

    def logits(self, x, head):
        d = self.model.depth_decoder
        return d.model.norm(x[:, -1, :]) @ d.codebooks_head.weight[head]

    def _function(self, temperature, top_p, top_k, cfg_scale, use_cfg):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_audio.lm.sample_utils import make_sampler

        key = (temperature, top_p, top_k, cfg_scale, use_cfg)
        if key in self.functions:
            return self.functions[key]
        valid = self.model.vocab_size
        effective_k = min(top_k, valid) if top_k else 0
        if effective_k == valid:
            effective_k = 0
        sample = make_sampler(temp=temperature, top_p=top_p, top_k=effective_k)

        def frame(first, hidden):
            x, caches = self.initial(first, hidden)
            tokens = [first.reshape(1)]
            for head in range(self.model.num_codebooks - 1):
                scores = self.logits(x, head)
                if use_cfg:
                    scores = scores[1:2] + cfg_scale * (scores[:1] - scores[1:2])
                scores = self.model._mask_reserved_codec_logits(scores)[..., :valid]
                token = sample(nn.log_softmax(scores, axis=-1)).astype(mx.int32).reshape(1)
                tokens.append(token)
                if head + 1 < self.model.num_codebooks - 1:
                    x = self.advance(token, head + 1, caches, hidden.shape[0])
            return mx.concatenate(tokens)

        fn = (mx.compile(frame, inputs=mx.random.state, outputs=mx.random.state)
              if self.mode == "compiled" else frame)
        self.functions[key] = fn
        return fn

    def generate_frame(self, first_codebook, conditional_hidden, *,
                       unconditional_hidden, cfg_scale, temperature, top_p, top_k):
        import mlx.core as mx
        if conditional_hidden.shape[0] != 1:
            raise ValueError("FastDepth currently supports one utterance at a time.")
        if self.model.num_codebooks != self.model.depth_decoder.model.num_codebooks:
            raise ValueError("Wrapper/depth codebook counts do not match.")
        hidden = conditional_hidden
        use_cfg = unconditional_hidden is not None
        if use_cfg:
            hidden = mx.concatenate((hidden, unconditional_hidden), axis=0)
        fn = self._function(temperature, top_p, top_k, cfg_scale, use_cfg)
        codes = fn(mx.array([first_codebook], dtype=mx.int32), hidden)
        # One host read for all remaining codebooks instead of .item() per code.
        return codes.tolist()

    def install(self):
        if self.mode == "native":
            return
        target = self.model
        cls = type(target)
        original = cls._depth_tokens
        self.original = original
        self.owned_method = "_depth_tokens" in cls.__dict__

        def optimized(obj, *args, **kwargs):
            if obj is target:
                return self.generate_frame(*args, **kwargs)
            return original(obj, *args, **kwargs)

        cls._depth_tokens = optimized

    def close(self):
        if self.original is not None:
            if self.owned_method:
                type(self.model)._depth_tokens = self.original
            else:
                delattr(type(self.model), "_depth_tokens")
            self.original = None

    def warmup(self, frames, cfg_scale, instruct):
        import mlx.core as mx
        m = self.model.depth_decoder.model
        try:
            # Shape/dtype match the projected backbone output. This warms depth
            # only; it is NOT a full text/codec warmup and is reported separately.
            hidden = mx.zeros((1, m.backbone_hidden_size),
                              dtype=self.model.depth_decoder.codebooks_head.weight.dtype)
            for _ in range(frames):
                self.model._depth_tokens(
                    1, hidden, unconditional_hidden=hidden if instruct and cfg_scale != 1 else None,
                    cfg_scale=cfg_scale, temperature=0.9, top_p=1.0, top_k=50)
        finally:
            mx.random.seed(0)  # generate(seed=args.seed) reseeds each measured run.


def verify_depth(model, engine):
    """Teacher-forced comparison of actual loaded weights, without sampling.

    Check all head indices and positional offsets over two independent frames.
    Quantized weights remain loaded; no full-size float model is instantiated.
    """
    import mlx.core as mx
    worst = 0.0
    try:
        mx.random.seed(20260906)
        m = model.depth_decoder.model
        for trial in range(2):
            hidden = mx.random.normal((1, m.backbone_hidden_size)).astype(
                model.depth_decoder.codebooks_head.weight.dtype)
            ids = [(3 + 17 * j + trial) % model.config.codec_vocab_size
                   for j in range(model.num_codebooks)]
            x, caches = engine.initial(mx.array([ids[0]], dtype=mx.int32), hidden)
            for head in range(model.num_codebooks - 1):
                expected = model.depth_decoder.next_logits(
                    mx.array([[0] + ids[:head+1]], dtype=mx.int32), hidden).astype(mx.float32)
                actual = engine.logits(x, head).astype(mx.float32)
                mx.eval(expected, actual)
                error = float(mx.max(mx.abs(actual - expected)).item())
                scale = max(float(mx.max(mx.abs(expected)).item()), 1e-6)
                worst = max(worst, error / scale)
                # BF16/4bit matrix-vector kernels can differ from prefix GEMM.
                # A gross mismatch stops execution; this is not an audio test.
                if not bool(mx.all(mx.isfinite(actual)).item()) or error > 0.03 * scale + 1e-4:
                    raise RuntimeError(f"Depth verification failed at head {head}: error={error}, scale={scale}")
                if head + 1 < model.num_codebooks - 1:
                    x = engine.advance(mx.array([ids[head+1]], dtype=mx.int32),
                                       head+1, caches, 1)
    finally:
        mx.random.seed(0)  # generate(seed=args.seed) reseeds each measured run.
    print(f"已加载权重的 depth 检查通过：最大绝对误差 / 对应 logits 最大幅值 = {worst:.3%}", flush=True)
    return worst


def pcm16_bytes(pcm):
    """The exact same clipped PCM16 representation for playback and WAV."""
    import numpy as np
    return (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()


def playback_worker(audio_q, status_q, rate, prebuffer_ms, latency_ms, device):
    """Fresh spawn process: no MLX model, no Python PortAudio callback.

    Blocking PortAudio writes run here, independently of the inference GIL.
    The parent queue is unbounded so audio-device pacing does not throttle
    synthesis. PCM ordering and EOF are supplied by one producer.
    """
    stream = None
    try:
        import sounddevice as sd
        stream = sd.RawOutputStream(
            samplerate=rate, channels=1, dtype="int16", blocksize=0,
            latency=latency_ms / 1000, device=device, dither_off=True,
        )
        status_q.put(("ready", {
            "device": sd.query_devices(stream.device)["name"],
            "sample_rate": float(stream.samplerate),
            "actual_latency_ms": float(stream.latency) * 1000,
        }))
        threshold = max(1, math.ceil(rate * prebuffer_ms / 1000)) * 2
        pending, buffered = [], 0
        eof = False
        while buffered < threshold:
            data = audio_q.get()
            if data is None:
                eof = True
                break
            pending.append(data)
            buffered += len(data)
        stats = {"device_underflows": 0, "startup_underflows": 0,
                 "first_audio_submit_perf": None, "queue_wait_ms": 0.0,
                 "written_samples": 0}
        if pending:
            stream.start()

            def write(data):
                first = stats["first_audio_submit_perf"] is None
                if first:
                    stats["first_audio_submit_perf"] = time.perf_counter()
                underflow = stream.write(data)
                if underflow:
                    stats["startup_underflows" if first else "device_underflows"] += 1
                stats["written_samples"] += len(data) // 2

            # One initial write avoids gaps between the prebuffer's small blocks.
            write(b"".join(pending))
            while not eof:
                stamp = time.perf_counter()
                data = audio_q.get()
                stats["queue_wait_ms"] += (time.perf_counter() - stamp) * 1000
                if data is None:
                    break
                write(data)
            stream.stop()  # Drain final audio before closing the device.
        stream.close()
        stream = None
        status_q.put(("done", stats))
    except BaseException as exc:
        status_q.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if stream is not None:
            try:
                stream.abort()
            finally:
                stream.close()


class ProcessPlayer:
    """Nonblocking producer API; child owns the only audio-device stream."""
    def __init__(self, rate, prebuffer_ms=320, latency_ms=100, device=None):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")  # Never fork a process containing MLX/Metal.
        self.rate = rate
        self.audio_q, self.status_q = ctx.Queue(), ctx.Queue()
        self.done = False
        self.finished = False
        self.error = None
        self.stats = {}
        self.device_info = None
        self.first_dac = None  # Blocking writes do not provide DAC timestamps.
        self.gap_samples = None  # Device-inserted gap length is not observable here.
        self.device_underflows = 0
        self.process = ctx.Process(
            target=playback_worker,
            args=(self.audio_q, self.status_q, rate, prebuffer_ms, latency_ms, device),
            daemon=True,
        )
        self.process.start()
        try:
            kind, data = self.status_q.get(timeout=20)
            if kind != "ready":
                raise RuntimeError(f"播放器启动失败: {data}")
            self.device_info = data
        except BaseException:
            self.close()
            raise

    def _receive(self, message):
        kind, data = message
        if kind == "error":
            self.error = data
        elif kind == "done":
            self.stats = data
            self.device_underflows = data["device_underflows"]
            self.done = True

    def _check(self):
        while True:
            try:
                self._receive(self.status_q.get_nowait())
            except queue.Empty:
                break
        if self.error:
            raise RuntimeError(f"播放进程错误: {self.error}")
        if not self.done and self.process.exitcode is not None:
            # The final status can arrive just after exitcode becomes visible.
            try:
                self._receive(self.status_q.get(timeout=0.1))
            except queue.Empty:
                pass
            if self.error or not self.done:
                raise RuntimeError(f"播放进程退出: {self.error or self.process.exitcode}")

    def put_bytes(self, data):
        if self.finished:
            raise RuntimeError("Cannot queue audio after EOF.")
        if len(data) % 2:
            raise ValueError("PCM16 data must contain complete samples.")
        self._check()
        if data:
            self.audio_q.put(data)

    def put(self, pcm):
        self.put_bytes(pcm16_bytes(pcm))

    def finish(self):
        if not self.finished:
            self._check()
            self.audio_q.put(None)
            self.finished = True

    def wait(self, timeout):
        deadline = time.perf_counter() + timeout
        while not self.done:
            self._check()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return False
            if not self.done:
                try:
                    self._receive(self.status_q.get(timeout=min(remaining, 0.2)))
                except queue.Empty:
                    pass
        return True

    def close(self):
        if self.done:
            self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
        for q in (self.audio_q, self.status_q):
            q.cancel_join_thread()
            q.close()


def make_player(rate, args):
    if args.no_play:
        return None
    if args.player == "callback":
        return Player(rate, args.prebuffer_ms, args.audio_latency_ms, args.device)
    return ProcessPlayer(rate, args.prebuffer_ms, args.audio_latency_ms, args.device)


def replay_wav(args):
    """Use the same audio path with no model/inference for diagnosis."""
    import numpy as np
    with wave.open(str(Path(args.play_wav).expanduser()), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise SystemExit("--play-wav 支持本脚本生成的单声道 PCM16 WAV。")
        rate, samples = wav.getframerate(), wav.getnframes()
        player = make_player(rate, args)
        try:
            while data := wav.readframes(max(1, rate // 5)):
                if isinstance(player, ProcessPlayer):
                    player.put_bytes(data)
                else:
                    player.put(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768)
            player.finish()
            if not player.wait(samples / rate + 15):
                raise RuntimeError("WAV 回放超时。")
            print(f"WAV 回放完成；设备 underflow: {player.device_underflows}")
            if isinstance(player, ProcessPlayer):
                print(f"播放设备: {player.device_info['device']}")
        finally:
            player.close()


class Player:
    """One producer; callback consumes PCM and an ordered EOF sentinel."""

    def __init__(self, rate, prebuffer_ms=320, latency_ms=100, device=None):
        import sounddevice as sd

        self.sd = sd
        self.rate = rate
        self.threshold = math.ceil(rate * prebuffer_ms / 1000)
        self.q = queue.SimpleQueue()
        self.current = None
        self.offset = 0
        self.submitted = 0
        self.running = False
        self.first_dac = None
        self.gap_samples = 0
        self.device_underflows = 0
        self.done = threading.Event()
        self.stream = sd.OutputStream(
            samplerate=rate, channels=1, dtype="float32", blocksize=0,
            latency=latency_ms / 1000, device=device, callback=self.callback, finished_callback=self.done.set,
        )

    def _start(self):
        if not self.running:
            self.stream.start()
            self.running = True

    def put(self, pcm):
        if not len(pcm):
            return
        self.q.put(pcm)
        self.submitted += len(pcm)
        if self.submitted >= self.threshold:
            self._start()

    def finish(self):
        # EOF cannot overtake queued audio, even if the last model chunk has
        # is_final_chunk=False (normal for exact multiples in mlx-audio 0.5.1).
        self.q.put(None)
        self._start()  # Also flush utterances shorter than the prebuffer.

    def callback(self, out, frames, timing, status):
        out.fill(0)
        if status.output_underflow and self.first_dac is not None:
            self.device_underflows += 1
        pos = 0
        while pos < frames:
            if self.current is None:
                try:
                    self.current = self.q.get_nowait()
                except queue.Empty:
                    if self.first_dac is not None:
                        self.gap_samples += frames - pos
                    return
                if self.current is None:
                    # CallbackStop drains already submitted output, unlike abort.
                    raise self.sd.CallbackStop
                self.offset = 0
            if self.first_dac is None:
                self.first_dac = (time.perf_counter()
                    + timing.outputBufferDacTime - timing.currentTime
                    + pos / self.rate)
            n = min(frames - pos, len(self.current) - self.offset)
            out[pos:pos+n, 0] = self.current[self.offset:self.offset+n]
            pos += n
            self.offset += n
            if self.offset == len(self.current):
                self.current = None

    def wait(self, timeout):
        return self.done.wait(timeout)

    def close(self):
        try:
            if self.running and not self.done.is_set():
                self.stream.abort()
        finally:
            self.stream.close()


def pcm_chunks(model, args):
    """Yield fresh noncumulative PCM, materialized before crossing threads."""
    import mlx.core as mx
    import numpy as np

    decoder = model.audio_tokenizer.decoder
    results = model.generate(
        text=args.text, instruct=args.instruct or None, cfg_scale=args.cfg_scale,
        max_tokens=args.max_tokens, seed=args.seed, stream=True,
        streaming_interval=args.streaming_interval,
    )
    try:
        for result in results:
            if not result.is_streaming_chunk:
                raise RuntimeError("Breeze 未返回流式块，请检查依赖版本。")
            if int(result.sample_rate) != int(model.sample_rate):
                raise RuntimeError("流中采样率发生变化。")
            if int(result.token_count) <= 0:
                # Upstream can return dummy audio on immediate EOS.
                raise RuntimeError("模型未生成声学帧（立即 EOS），请调整文本或 seed。")
            pcm = np.array(result.audio.astype(mx.float32), copy=True)
            if pcm.ndim != 1 or not np.isfinite(pcm).all():
                raise RuntimeError("模型返回无效 PCM。")
            if len(pcm) == 0:
                raise RuntimeError("解码器返回空 PCM。")
            yield pcm, int(result.token_count), bool(result.is_final_chunk)
    finally:
        # Native state resets on normal completion; also reset on cancellation.
        try:
            results.close()
        finally:
            decoder.reset_streaming_state()


def benchmark(model, args, run_number, load_s):
    import mlx.core as mx
    import numpy as np

    rate = int(model.sample_rate)
    player = make_player(rate, args)
    profiler = StageProfiler(model, args.profile)
    parts, arrivals = [], []
    total_samples = total_frames = 0
    print(f"\nRun {run_number}: {'首次推理' if run_number == 1 else '同进程重复推理'}", flush=True)
    mx.synchronize()
    start = time.perf_counter()
    chunks = pcm_chunks(model, args)
    try:
        profiler.install()
        for pcm, frames, final in chunks:
            stamp = time.perf_counter()
            total_samples += len(pcm)
            total_frames += frames
            parts.append(pcm)
            arrivals.append({"elapsed_s": stamp - start, "samples": len(pcm),
                             "code_frames": frames, "final_flush": final})
            if player:
                player.put(pcm)
            if len(arrivals) == 1:
                print(f"首块 PCM: {(stamp-start)*1000:.1f} ms", flush=True)
        mx.synchronize()
        elapsed = time.perf_counter() - start
        if not total_samples:
            raise RuntimeError("未生成音频。")
        if player:
            player.finish()
            if not player.wait(total_samples / rate + 15):
                raise RuntimeError("播放超时；可使用 --no-play 单独测试生成。")
    finally:
        try:
            chunks.close()
        finally:
            profiler.close()
            if player:
                player.close()

    duration = total_samples / rate
    rtf = elapsed / duration
    out = Path(f"{args.output}_{run_number}").expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.concatenate(parts)
    pcm16 = pcm16_bytes(waveform)
    with wave.open(str(out) + ".wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm16)
    report = {
        "backend": f"Breeze TTS 2 / {args.depth_mode} depth + native MLX streaming codec",
        "depth_mode": args.depth_mode, "depth_warmup_s": args.depth_warmup_s,
        "warmup_frames": args.warmup_frames,
        "model": args.model, "mlx_audio": version("mlx-audio"),
        "mlx": version("mlx"), "platform": platform.platform(),
        "machine": platform.machine(), "run": run_number,
        "hardware": getattr(args, "hardware", None),
        "profiling_enabled": args.profile,
        "stage_timings": profiler.stats if args.profile else None,
        "profile_other_s": (max(0.0, elapsed - sum(
            item["seconds"] for item in profiler.stats.values())) if args.profile else None),
        "text": args.text, "instruct": args.instruct, "cfg_scale": args.cfg_scale,
        "seed": args.seed, "sample_rate": rate, "load_s": load_s,
        "chunk_frames": args.chunk_frames,
        "streaming_interval_s": args.streaming_interval,
        "prebuffer_ms": args.prebuffer_ms, "max_tokens": args.max_tokens,
        "player_backend": args.player if player else None,
        "audio_latency_requested_ms": args.audio_latency_ms,
        "playback_device": player.device_info if isinstance(player, ProcessPlayer) else None,
        "raw_pcm_peak_abs": float(np.max(np.abs(waveform))),
        "raw_pcm_out_of_range_samples": int(np.count_nonzero(np.abs(waveform) > 1)),
        "playback_stats": player.stats if isinstance(player, ProcessPlayer) else None,
        "first_audio_submit_ms": ((player.stats["first_audio_submit_perf"] - start) * 1000
            if isinstance(player, ProcessPlayer) and player.stats.get("first_audio_submit_perf") is not None else None),
        "generated_frames": total_frames,
        "hit_max_tokens": total_frames >= args.max_tokens,
        "ttfa_ms": arrivals[0]["elapsed_s"] * 1000,
        "generation_s": elapsed, "audio_s": duration, "rtf": rtf,
        "rtf_below_one": rtf < 1, "pcm_chunks": len(arrivals),
        "multiple_pcm_chunks": len(arrivals) > 1,
        "first_pcm_before_final_flush": not arrivals[0]["final_flush"],
        "first_dac_estimate_ms": ((player.first_dac-start)*1000
            if player and player.first_dac is not None else None),
        "playback_gap_ms": player.gap_samples / rate * 1000 if player and player.gap_samples is not None else None,
        "device_underflows": player.device_underflows if player else None,
        "mlx_process_peak_gb": mx.get_peak_memory() / 1e9,
        "arrivals": arrivals,
        "measurement_notes": [
            "TTFA includes prompt preparation and first PCM materialization; excludes model load.",
            "RTF uses wall time through iterator exhaustion, including wrapper overhead; excludes playback drain and file writes.",
            "DAC time is PortAudio's estimate, not acoustic onset; peak MLX memory is process-wide, not total system RAM.",
            "Callback queue gaps measure inserted silence. Process mode has no exact gap/DAC measurement: null means unavailable, not zero.",
            "Process first_audio_submit_ms measures first blocking write submission, not acoustic onset. queue_wait_ms includes waits that device buffers may cover, so it is not audible gap duration.",
            "Process mode plays the same clipped PCM16 samples saved to WAV. Spawned device startup is completed before TTFA/RTF timing.",
            "A missing final flag is valid on exact chunk multiples; iterator exhaustion defines EOF.",
            "Profiling adds GPU synchronization; stage times diagnose bottlenecks but perturb normal throughput. Other time includes sampling, Python work, synchronization before stages, and wrapper overhead.",
        ],
    }
    Path(str(out) + ".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"合成 {elapsed:.3f}s / 音频 {duration:.3f}s = RTF {rtf:.3f}；RTF < 1: {rtf < 1}")
    print(f"PCM 块数: {len(arrivals)}；WAV/JSON: {out}")
    if args.profile:
        print("分阶段诊断（额外 GPU 同步会影响速度）：")
        labels = {"prompt": "提示处理", "backbone_prefill": "主干预填充",
                  "backbone_decode": "主干逐帧", "depth": "深度解码",
                  "codec": "音频解码"}
        for name, item in profiler.stats.items():
            seconds, calls = item["seconds"], item["calls"]
            print(f"  {labels[name]}: {seconds:.3f}s / {calls} 次 / "
                  f"平均 {seconds/calls*1000:.2f}ms / 总耗时占比 {seconds/elapsed:.1%}")
        print(f"  其他（含采样、Python 及同步开销）: {report['profile_other_s']:.3f}s")
    if isinstance(player, ProcessPlayer):
        print(f"播放设备: {player.device_info['device']}；"
              f"实际设备延迟: {player.device_info['actual_latency_ms']:.1f}ms")
        first_submit = report["first_audio_submit_ms"]
        print(f"首批音频提交: {first_submit:.1f}ms；"
              f"持续播放 underflow: {player.device_underflows}；"
              f"启动 underflow: {player.stats['startup_underflows']}")
        if player.stats["written_samples"] != total_samples:
            raise RuntimeError("播放采样点数与 WAV 不一致。")
    elif player:
        print(f"首块 DAC 估计: {report['first_dac_estimate_ms']:.1f}ms；"
              f"播放缺音: {report['playback_gap_ms']:.1f}ms；"
              f"设备 underflow: {player.device_underflows}")
    if report["raw_pcm_out_of_range_samples"]:
        print(f"原始 PCM 有 {report['raw_pcm_out_of_range_samples']} 个越界采样点，"
              "独立进程播放和 WAV 均已做相同限幅。")
    if len(arrivals) == 1:
        print("本轮只有一块 PCM；请加长文本或减小 --chunk-frames 验证多块输出。")
    if report["hit_max_tokens"]:
        print("已达到 --max-tokens，语音可能被截断；可提高上限后重试。")
    if rtf >= 1:
        print("本轮尚未达到实时；预缓冲只能推迟缺音。")
        if args.depth_mode == "cached":
            print("可用相同文本比较 --depth-mode compiled；首次编译耗时单独记录。")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", default="你好，我现在正在这台麦克电脑上进行完全本地的流式语音合成，请听一下第一段声音出现需要多久，以及后面的播放是否流畅。")
    parser.add_argument("--model", default="mlx-community/Breeze-TTS-2-mlx-4bit")
    parser.add_argument("--instruct", "--instruction", default="一位声音自然清晰、语气亲切的中文说话者，以正常语速说话。")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--chunk-frames", type=int, default=2)
    parser.add_argument("--prebuffer-ms", type=float, default=320)
    parser.add_argument("--max-tokens", type=int, default=750)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="breeze_fast")
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--player", choices=["process", "callback"], default="process",
                        help="默认独立播放进程；callback 保留旧回调路径用于对照")
    parser.add_argument("--audio-latency-ms", type=float, default=100)
    parser.add_argument("--device", help="输出设备编号或名称；省略使用系统默认设备")
    parser.add_argument("--list-audio-devices", action="store_true")
    parser.add_argument("--play-wav", help="仅回放已有 WAV，使用同一播放链路，不加载模型")
    parser.add_argument("--profile", action="store_true",
                        help="开启分阶段诊断；额外 GPU 同步会影响测速，建议配合 --no-play")
    parser.add_argument("--depth-mode", choices=["native", "cached", "compiled"], default="cached")
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--verify-depth", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.audio_latency_ms) or args.audio_latency_ms <= 0:
        parser.error("audio-latency-ms 必须是正的有限数值。")
    if args.device is not None and args.device.isdecimal():
        args.device = int(args.device)
    if args.play_wav and args.no_play:
        parser.error("--play-wav 与 --no-play 不能同时使用。")
    if args.warmup_frames < 0:
        parser.error("warmup-frames 不可为负数。")
    if not args.text.strip():
        parser.error("text 不能为空。")
    if min(args.chunk_frames, args.max_tokens, args.runs) < 1:
        parser.error("chunk-frames/max-tokens/runs 必须为正整数。")
    if not math.isfinite(args.prebuffer_ms) or args.prebuffer_ms < 0:
        parser.error("prebuffer-ms 必须是非负有限数值。")
    if not math.isfinite(args.cfg_scale):
        parser.error("cfg-scale 必须是有限数值。")
    return args


def main():
    args = parse_args()
    if args.list_audio_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.play_wav:
        replay_wav(args)
        return
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("需要 Apple Silicon Mac 和原生 arm64 Python（不支持 Intel/Rosetta）。")
    if version("mlx") != "0.32.2":
        raise SystemExit("本脚本固定使用 mlx==0.32.2，请用顶部 uv 命令运行。")
    if version("mlx-audio") != "0.5.1":
        raise SystemExit("此脚本核对的版本为 mlx-audio==0.5.1，请使用顶部 uv 命令运行。")
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.models.breeze_tts.breeze_tts import Model

    if not mx.metal.is_available():
        raise SystemExit("MLX Metal GPU 不可用。")
    args.hardware = hardware_info()
    print(f"设备: {args.hardware['chip']}；物理内存: {args.hardware['memory_gib']} GiB", flush=True)
    if args.profile:
        print("已开启分阶段诊断；本轮速度受额外 GPU 同步影响。", flush=True)
    print(f"加载 {args.model}；首次需要下载，加载时间单独记录。", flush=True)
    started = time.perf_counter()
    model = load_model(args.model)
    if not isinstance(model, Model):
        raise SystemExit("此脚本只支持 Breeze TTS 2 MLX 权重。")
    mx.synchronize()
    load_s = time.perf_counter() - started
    codec = model.audio_tokenizer
    rate = getattr(codec, "decode_upsample_rate", None)
    if rate is None:
        rate = getattr(codec.decoder, "decode_upsample_rate", None)
    if rate is None or rate <= 0:
        raise SystemExit("无法读取 codec 每帧采样点数。")
    # Upstream converts seconds back with int(...); move slightly inside the
    # requested frame bin to avoid floating-point truncation to n-1 frames.
    args.streaming_interval = (args.chunk_frames + 1e-6) * rate / model.sample_rate
    print(f"加载 {load_s:.2f}s；每块 {args.chunk_frames} 帧，"
          f"约 {args.chunk_frames*rate/model.sample_rate*1000:.0f}ms 音频；"
          f"CFG={args.cfg_scale}；原生句内流式。", flush=True)
    engine = FastDepth(model, args.depth_mode)
    if args.verify_depth:
        verify_depth(model, engine)
    try:
        engine.install()
        print(f"Depth 模式: {args.depth_mode}；预热 {args.warmup_frames} 帧。"
              + ("首次整帧编译可能较慢。" if args.depth_mode == "compiled" else ""), flush=True)
        warm_start = time.perf_counter()
        engine.warmup(args.warmup_frames, args.cfg_scale, args.instruct)
        mx.synchronize()
        args.depth_warmup_s = time.perf_counter() - warm_start
        print(f"Depth 预热: {args.depth_warmup_s:.3f}s（不计入 TTFA/RTF）", flush=True)
        print("播放: " + ("关闭（--no-play）；仍保存 WAV" if args.no_play else
              f"开启 / {args.player} / 预缓冲 {args.prebuffer_ms:g}ms / 设备延迟请求 {args.audio_latency_ms:g}ms"), flush=True)
        for run_number in range(1, args.runs + 1):
            benchmark(model, args, run_number, load_s)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
