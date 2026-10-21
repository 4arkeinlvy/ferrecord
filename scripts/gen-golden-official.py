#!/usr/bin/env python3
"""Reference Silero probabilities from the OFFICIAL model + onnxruntime.

Reproduces snakers4/silero-vad `OnnxWrapper.__call__` (utils_vad.py) exactly:
each 512-sample chunk is prefixed with the previous input's last 64 samples
(zeros at stream start), the LSTM state is carried across calls, and the
context is updated to the tail of the concatenated input.

Run inside:  nix-shell -p "python3.withPackages (ps: [ ps.onnxruntime ps.numpy ])"
Output:      testdata/fixtures/silero_v6.2.3_ctx_aepyx_16k.json  (committed fixture)
"""

import hashlib
import json
import os
import sys
import urllib.request
import wave

import numpy as np
import onnxruntime as ort

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
MODEL = os.path.join(ROOT, "testdata/golden/silero_vad_v6.2.3.onnx")
AUDIO = os.path.join(ROOT, "testdata/golden/aepyx.wav")  # fetched by scripts/fetch-golden.sh
OUT = os.path.join(ROOT, "testdata/fixtures/silero_v6.2.3_ctx_aepyx_16k.json")
SR, CHUNK, CONTEXT = 16000, 512, 64


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_model():
    if not os.path.exists(MODEL) or sha256(MODEL) != MODEL_SHA256:
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    got = sha256(MODEL)
    if got != MODEL_SHA256:
        sys.exit(f"model checksum mismatch: {got}")


def read_pcm16_mono(path):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == SR, "need 16 kHz mono PCM16"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def main():
    ensure_model()
    audio = read_pcm16_mono(AUDIO)
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(MODEL, sess_options=opts, providers=["CPUExecutionProvider"])

    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, CONTEXT), dtype=np.float32)
    probs = []
    for i in range(len(audio) // CHUNK):
        x = audio[i * CHUNK:(i + 1) * CHUNK][None, :]
        x = np.concatenate([context, x], axis=1)  # OnnxWrapper: torch.cat([self._context, x], dim=1)
        out, state = sess.run(None, {"input": x, "state": state, "sr": np.array(SR, dtype=np.int64)})
        context = x[..., -CONTEXT:]
        probs.append(float(out[0][0]))

    doc = {
        "model": "snakers4/silero-vad v6.2.3 silero_vad.onnx",
        "model_sha256": MODEL_SHA256,
        "onnxruntime": ort.__version__,
        "audio": "aepyx.wav",
        "audio_sha256": sha256(AUDIO),
        "sample_rate": SR,
        "chunk_size": CHUNK,
        "context_size": CONTEXT,
        "probs": [round(p, 7) for p in probs],
    }
    with open(OUT, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
        f.write("\n")
    speech = sum(p >= 0.5 for p in probs)
    print(f"wrote {OUT}: {len(probs)} chunks, {speech} ({100 * speech / len(probs):.0f}%) >= 0.5")


if __name__ == "__main__":
    main()
