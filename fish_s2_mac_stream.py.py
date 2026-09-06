# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlx-audio==0.5.1", "sounddevice==0.5.3"]
# ///
"""Local S2 Pro / Apple Silicon streaming experiment.

Run: uv run --python 3.12 fish_s2_mac_stream.py
Offline after first successful run:
  HF_HUB_OFFLINE=1 uv run --offline --python 3.12 fish_s2_mac_stream.py

The 0.5.1 public Fish API does not implement stream=True. This script adds
an in-memory yield to its autoregressive loop and decodes new frames with cached
codec state. It does not modify package files or split the input text.
The first and subsequent PCM blocks default to 8 frames each.
The default codec now caches causal convolution history and transformer K/V.
Use --decoder prefix to compare against the old prefix-redecoding baseline.
This is an experimental adapter, not Fish's optimized server.
Numerical equivalence was checked on CPU MLX with small random-weight codecs,
including attention-window crossings and the production convolution strides.
Real-weight audio quality and speed have NOT been tested on an actual Mac.

Sources:
https://github.com/Blaizzy/mlx-audio
https://huggingface.co/mlx-community/fish-audio-s2-pro-8bit
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import platform
import queue
import textwrap
import threading
import time
import wave
from importlib.metadata import version
from pathlib import Path


def streaming_function(original):
    """Retain the release's sampling/cache logic; expose completed code frames."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    count = 0

    class InsertYield(ast.NodeTransformer):
        def visit_Expr(self, node):
            nonlocal count
            call = node.value
            if (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "generated_steps"
                    and call.func.attr == "append"):
                count += 1
                return [node, ast.Expr(ast.Yield(ast.Name("generated_steps", ast.Load())))]
            return node

        def visit_Return(self, node):
            # The caller already has all frames. Avoid a redundant final stack.
            return ast.Return(value=None)

    tree = InsertYield().visit(tree)
    if count != 1:
        raise RuntimeError("Fish code changed: expected exactly one frame append.")
    fn = tree.body[0]
    fn.decorator_list = []
    fn.returns = None
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, "<fish-stream-adapter>", "exec"), namespace)
    return namespace[fn.name]


class Player:
    """Audio-device callback drains an unbounded queue; producer never waits."""
    def __init__(self, rate):
        import numpy as np
        import sounddevice as sd
        self.np = np
        self.sd = sd
        self.rate = rate
        self.q = queue.SimpleQueue()
        self.current = None
        self.offset = 0
        self.started = False
        self.eof = False
        self.done = threading.Event()
        self.first_dac = None
        self.gap_samples = 0
        self.device_underflows = 0
        self.stream = sd.OutputStream(
            samplerate=rate, channels=1, dtype="float32", blocksize=0,
            latency="low", callback=self.callback, finished_callback=self.done.set,
        )

    def callback(self, out, frames, timing, status):
        out.fill(0)
        if status.output_underflow and self.started:
            self.device_underflows += 1
        pos = 0
        while pos < frames:
            if self.current is None:
                try:
                    self.current = self.q.get_nowait()
                    self.offset = 0
                except queue.Empty:
                    if self.eof:
                        raise self.sd.CallbackStop
                    if self.started:
                        self.gap_samples += frames - pos
                    return
            if not self.started:
                self.started = True
                # PortAudio's predicted DAC submission time, not acoustic onset.
                self.first_dac = (time.perf_counter()
                    + timing.outputBufferDacTime - timing.currentTime
                    + pos / self.rate)
            n = min(frames - pos, len(self.current) - self.offset)
            out[pos:pos+n, 0] = self.current[self.offset:self.offset+n]
            pos += n
            self.offset += n
            if self.offset == len(self.current):
                self.current = None


class StatefulCodec:
    """Incremental adapter for the causal Fish S1 DAC in mlx-audio 0.5.1.

    Keeps the original layers/weights and finite convolution history. Transformer
    K/V caches retain the exact attention window for each layer. No text/audio
    context is approximated by arbitrary waveform cropping.
    """
    def __init__(self, codec):
        import mlx.core as mx
        from mlx_audio.codec.models.fish_s1_dac import fish_s1_dac as dac
        self.mx, self.dac, self.codec = mx, dac, codec
        self.history = {}
        self.kv = {}
        self.offsets = {}
        self.frozen = {}
        # Weight normalization is invariant during inference. Materialize once.
        for _, layer in codec.decoder.named_modules():
            if isinstance(layer, (dac.CausalWNConv1d, dac.CausalWNConvTranspose1d)):
                raw = layer.conv
                raw.conv.weight = (layer.weight_g * layer.weight_v
                    / dac._normalize_weight(layer.weight_v, except_dim=0))
                raw.conv.bias = layer.bias
                mx.eval(raw.conv.weight)
                self.frozen[id(layer)] = raw

    def _conv(self, layer, x):
        mx, dac = self.mx, self.dac
        transposed = isinstance(layer, dac.CausalTransConvNet)
        if not transposed and layer.stride != 1:
            raise RuntimeError("Incremental decoder only supports stride-one causal convolutions.")
        # Include enough earlier input to reproduce the new output's entire
        # receptive field, then discard the recalculated history output.
        keep = ((layer.kernel_size - 1 + layer.stride - 1) // layer.stride
                if transposed else layer.kernel_size - 1)
        old = self.history.get(id(layer))
        joined = x if old is None else mx.concatenate([old, x], axis=-1)
        if keep:
            self.history[id(layer)] = mx.contiguous(joined[..., -keep:])
        count = x.shape[-1] * (layer.stride if transposed else 1)
        return layer(joined)[..., -count:]

    def _transformer(self, module, x):
        mx, dac = self.mx, self.dac
        if not module.causal or not isinstance(module.look_ahead_conv, dac.Identity):
            raise RuntimeError("Unsupported noncausal/look-ahead codec transformer.")
        if module.channels_first:
            x = x.swapaxes(1, 2)
        x = module.input_proj(x)
        length = x.shape[1]
        offset = self.offsets.get(id(module), 0)
        if offset + length > module.config.block_size:
            raise RuntimeError("Codec positional limit exceeded; shorten the text.")
        freqs = None if module.freqs_cis is None else module.freqs_cis[offset:offset+length]
        for block in module.layers:
            a = block.attention
            if a.n_head != a.n_local_heads:
                raise RuntimeError("Unsupported codec grouped-query configuration.")
            h = block.attention_norm(x)
            packed = a.wqkv(h)
            width = a.n_head * a.head_dim
            heads = [packed[..., i*width:(i+1)*width].reshape(
                x.shape[0], length, a.n_head, a.head_dim) for i in range(3)]
            q, k, v = heads
            if a.pos_embed_type == "rope" and freqs is not None:
                q, k = dac.apply_rotary_emb(q, freqs), dac.apply_rotary_emb(k, freqs)
            q, k, v = [t.transpose(0, 2, 1, 3) for t in (q, k, v)]
            old = self.kv.get(id(block))
            previous = 0 if old is None else old[0].shape[2]
            if old is not None:
                k, v = mx.concatenate([old[0], k], axis=2), mx.concatenate([old[1], v], axis=2)
            query_positions = mx.arange(offset, offset+length)[:, None]
            key_positions = mx.arange(offset-previous, offset+length)[None, :]
            allowed = key_positions <= query_positions
            if module.window_size:
                allowed = allowed & (key_positions >= query_positions-module.window_size+1)
            mask = mx.where(allowed, 0.0, -1e9).astype(x.dtype)[None, None]
            attended = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=a.head_dim**-0.5, mask=mask)
            attended = attended.transpose(0, 2, 1, 3).reshape(x.shape[0], length, width)
            h = x + block.attention_layer_scale(a.wo(attended))
            x = h + block.ffn_layer_scale(block.feed_forward(block.ffn_norm(h)))
            keep = module.window_size - 1 if module.window_size else k.shape[2]
            self.kv[id(block)] = (mx.contiguous(k[:, :, -keep:]) if keep else k[:, :, :0],
                                  mx.contiguous(v[:, :, -keep:]) if keep else v[:, :, :0])
        self.offsets[id(module)] = offset + length
        x = module.output_proj(module.norm(x))
        return x.swapaxes(1, 2) if module.channels_first else x

    def _apply(self, layer, x):
        dac = self.dac
        if id(layer) in self.frozen:
            return self._conv(self.frozen[id(layer)], x)
        if isinstance(layer, (dac.CausalConvNet, dac.CausalTransConvNet)):
            return self._conv(layer, x)
        if isinstance(layer, dac.WindowLimitedTransformer):
            return self._transformer(layer, x)
        if isinstance(layer, dac.ConvNeXtBlock):
            h = self._apply(layer.dwconv, x).swapaxes(1, 2)
            h = layer.pwconv2(layer.act(layer.pwconv1(layer.norm(h))))
            if layer.gamma is not None:
                h = h * layer.gamma
            return x + h.swapaxes(1, 2)
        if isinstance(layer, (dac.DecoderBlock, dac.ResidualUnit)):
            h = x
            for child in layer.block:
                h = self._apply(child, h)
            return x + h if isinstance(layer, dac.ResidualUnit) else h
        # Pointwise activations and identity have no temporal state.
        if type(layer).__name__ not in ("Snake1d", "Tanh", "Identity"):
            raise RuntimeError(f"Unsupported codec layer: {type(layer).__name__}")
        return layer(x)

    def decode(self, codes):
        mx = self.mx
        q = self.codec.quantizer
        ids = codes[None].astype(mx.int32)
        semantic = mx.clip(ids[:, :1], 0, q.semantic_quantizer.codebook_size-1)
        z = q.semantic_quantizer.from_codes(semantic)[0]
        if ids.shape[1] > 1:
            residual = mx.clip(ids[:, 1:], 0, q.quantizer.codebook_size-1)
            z = z + q.quantizer.from_codes(residual)[0]
        z = self._apply(q.post_module, z)
        for block in q.upsample:
            for layer in block:
                z = self._apply(layer, z)
        for layer in self.codec.decoder.model:
            z = self._apply(layer, z)
        states = list(self.history.values())
        for pair in self.kv.values():
            states.extend(pair)
        mx.eval(z, *states)
        expected = codes.shape[1] * self.codec.frame_length
        if z.shape[-1] != expected:
            raise RuntimeError(f"Unexpected incremental codec length: {z.shape[-1]} != {expected}")
        return z[0, 0]


def pcm_chunks(model, codes_fn, text, chunk_frames, hold_frames, max_tokens,
               first_frames=None, timings=None, codec_engine=None):
    import mlx.core as mx
    import numpy as np
    from mlx_audio.tts.models.fish_qwen3_omni.prompt import Message, TextPart

    timings = timings if timings is not None else {}
    timings.update(ar_s=0.0, codec_s=0.0, decoder_calls=0)
    ar_start = time.perf_counter()
    conversation = model._build_conversation([], [])
    conversation.append(Message(role="user", parts=[TextPart(text)],
                                add_im_start=True, add_im_end=True))
    emitted = 0
    frame_samples = int(model.codec.frame_length)
    next_decode = (first_frames or chunk_frames) + hold_frames
    last_decoded_frames = 0
    last_audio = None
    steps = None

    def decode_steps(current_steps):
        if codec_engine is None:
            codes = mx.stack(current_steps, axis=1).astype(mx.int32)
            return np.array(model._decode_codes(codes).astype(mx.float32))
        codes = mx.stack(current_steps[last_decoded_frames:], axis=1).astype(mx.int32)
        added = np.array(codec_engine.decode(codes).astype(mx.float32))
        return added if last_audio is None else np.concatenate([last_audio, added])

    for steps in codes_fn(model, conversation=conversation, batch_text=text,
                         max_new_tokens=max_tokens, top_p=0.7,
                         top_k=30, temperature=0.7):
        if len(steps) < next_decode:
            continue
        mx.synchronize()
        timings['ar_s'] += time.perf_counter() - ar_start
        decode_start = time.perf_counter()
        # Materialize PCM before handing it to the benchmark or sound device.
        last_audio = decode_steps(steps)
        timings['codec_s'] += time.perf_counter() - decode_start
        timings['decoder_calls'] += 1
        last_decoded_frames = len(steps)
        end = max(0, len(last_audio) - hold_frames * frame_samples)
        if end > emitted:
            yield last_audio[emitted:end].copy(), len(steps), False
            emitted = end
        next_decode += chunk_frames
        ar_start = time.perf_counter()
    mx.synchronize()
    timings['ar_s'] += time.perf_counter() - ar_start
    if not steps:
        raise RuntimeError("No acoustic frames generated.")
    if last_decoded_frames != len(steps):
        decode_start = time.perf_counter()
        last_audio = decode_steps(steps)
        timings['codec_s'] += time.perf_counter() - decode_start
        timings['decoder_calls'] += 1
    if len(last_audio) > emitted:
        yield last_audio[emitted:].copy(), len(steps), True


def benchmark(model, fn, args, run_number):
    import mlx.core as mx
    import numpy as np

    codec_engine = StatefulCodec(model.codec) if args.decoder == "cached" else None
    player = None if args.no_play else Player(int(model.sample_rate))
    audio_parts, arrivals = [], []
    total_samples = 0
    timings = {}
    mx.random.seed(args.seed)
    if player:
        player.stream.start()
    mx.synchronize()
    print(f"\nRun {run_number}: {'首次推理' if run_number == 1 else '同进程重复推理'}", flush=True)
    start = time.perf_counter()
    try:
        for pcm, token_count, final in pcm_chunks(
                model, fn, args.text, args.chunk_frames, args.hold_frames, args.max_tokens,
                first_frames=args.first_frames, timings=timings, codec_engine=codec_engine):
            stamp = time.perf_counter()
            if pcm.ndim != 1 or not np.isfinite(pcm).all():
                raise RuntimeError("Invalid PCM returned by codec.")
            total_samples += len(pcm)
            audio_parts.append(pcm)
            arrivals.append({"elapsed_s": stamp-start, "samples": len(pcm),
                             "code_frames": token_count, "final_flush": final})
            if player:
                player.q.put(pcm)
            if len(arrivals) == 1:
                print(f"首块 PCM: {(stamp-start)*1000:.1f} ms", flush=True)
        mx.synchronize()
        elapsed = time.perf_counter() - start
        if not total_samples:
            raise RuntimeError("Empty audio.")
        # End the stopwatch BEFORE draining the playback queue or writing files.
        if player:
            player.eof = True
            if not player.done.wait(total_samples / model.sample_rate + 10):
                raise RuntimeError("Audio playback timed out.")
    finally:
        if player:
            player.stream.abort()
            player.stream.close()

    duration = total_samples / model.sample_rate
    rtf = elapsed / duration
    out = Path(f"{args.output}_{run_number}")
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(np.concatenate(audio_parts), -1, 1) * 32767).astype("<i2")
    with wave.open(str(out) + ".wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(model.sample_rate))
        f.writeframes(pcm16.tobytes())
    report = {
        "backend": f"experimental Fish S2 Pro MLX / {args.decoder} codec",
        "model": args.model, "mlx_audio": version("mlx-audio"),
        "mlx": version("mlx"), "platform": platform.platform(),
        "run": run_number, "text": args.text, "seed": args.seed,
        "first_frames": args.first_frames,
        "chunk_frames": args.chunk_frames, "hold_frames": args.hold_frames,
        "ttfa_ms": arrivals[0]["elapsed_s"]*1000,
        "generation_s": elapsed, "audio_s": duration, "rtf": rtf,
        "rtf_below_one": rtf < 1, "pcm_chunks": len(arrivals),
        "first_pcm_before_final_flush": not arrivals[0]["final_flush"],
        "first_dac_estimate_ms": ((player.first_dac-start)*1000
            if player and player.first_dac is not None else None),
        "playback_gap_ms": player.gap_samples / model.sample_rate * 1000 if player else None,
        "device_underflows": player.device_underflows if player else None,
        "timings": timings,
        "ar_rtf": timings['ar_s'] / duration,
        "codec_rtf": timings['codec_s'] / duration,
        "arrivals": arrivals,
    }
    Path(str(out) + ".json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"合成 {elapsed:.3f}s / 音频 {duration:.3f}s = RTF {rtf:.3f}；RTF < 1: {rtf < 1}")
    print(f"模型生成: {timings['ar_s']:.3f}s (RTF {report['ar_rtf']:.3f})；"
          f"音频解码: {timings['codec_s']:.3f}s (RTF {report['codec_rtf']:.3f})；"
          f"解码调用: {timings['decoder_calls']}")
    print(f"PCM 块数: {len(arrivals)}；WAV/JSON: {out}")
    if player:
        print(f"播放缺音: {report['playback_gap_ms']:.1f}ms；设备 underflow: {player.device_underflows}")
    if len(arrivals) == 1:
        print("本轮只输出一块音频，不能用来证明句内流式；请加长文本或减小 chunk-frames。")
    if report['ar_rtf'] >= 1:
        print("模型生成本身已慢于实时；调整播放器缓冲无法解决持续缺音。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="你好，我现在正在这台麦克电脑上进行完全本地的流式语音合成，请听一下第一段声音出现需要多久，以及后面的播放是否流畅。")
    parser.add_argument("--model", default="mlx-community/fish-audio-s2-pro-8bit")
    parser.add_argument("--decoder", choices=["cached", "prefix"], default="cached")
    parser.add_argument("--first-frames", type=int, default=8,
                        help="首块音频帧数，不含 hold-frames")
    parser.add_argument("--chunk-frames", type=int, default=8,
                        help="后续每块帧数；增大可摊薄调用开销，但增加块间间隔")
    parser.add_argument("--hold-frames", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="fish_bench")
    parser.add_argument("--no-play", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("需要 Apple Silicon Mac 和原生 arm64 Python。")
    if (args.first_frames < 1 or args.chunk_frames < 1 or args.hold_frames < 0
            or args.runs < 1 or args.max_tokens < 1):
        parser.error("first-frames/chunk-frames/runs/max-tokens 必须为正数，hold-frames 不可为负数。")
    if not args.text.strip():
        parser.error("text 不能为空。")
    if version("mlx-audio") != "0.5.1":
        parser.error("此适配器固定使用 mlx-audio==0.5.1。")
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model

    if not mx.metal.is_available():
        parser.error("MLX Metal GPU 不可用。")
    fn = streaming_function(Model._generate_codes_for_batch)
    print("加载 S2 Pro 8bit；首次需要下载权重，推理完全在本机。加载不计入 TTFA/RTF。", flush=True)
    model = load_model(args.model)
    if not isinstance(model, Model):
        parser.error("此脚本只接受 Fish S2 Pro MLX 权重。")
    mx.synchronize()
    print(f"实验性流式适配器，decoder={args.decoder}；不会调用未实现的 stream=True。", flush=True)
    for run_number in range(1, args.runs + 1):
        benchmark(model, fn, args, run_number)


if __name__ == "__main__":
    main()
