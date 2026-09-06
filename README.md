# BreezeTTS2_Mac_Streaming
Real-time Breeze-TTS-2 streaming on Apple Silicon — RTF from ~4 down to ~1. mlx-audio 0.5.1's depth decoder recomputes each frame's existing acoustic codes instead of reusing its KV cache. Adds intra-frame KV reuse, one CPU read-back per frame, an optional compiled whole-frame path, and RTF/underflow stats.
