# ferrecord

Lightweight desktop auto-recorder. So user 
signs in once; the app records each conversation by itself (start on sustained
speech with an 8 s pre-roll, stop after 30 s of silence) and uploads it into the
existing Siloam pipeline, where it waits for confirmation like a browser recording.
Rust, Windows first. The full design is in the approved plan.

**Status: M0 (setup + detection) complete.** There's no app yet. What exists is the
platform-independent core, the tools to measure detection, and the measured defaults.

## Detection defaults (measured, see `vad-eval`)
- **Engine:** Silero VAD v6.2, pure Rust (vendored, context-patched, matches official onnxruntime to 2e-6).
- **Front end:** 90 Hz high-pass → RNNoise (`nnnoiseless`) at 48 kHz → 16 kHz.
- **Start:** 1 s of speech within 2 s. **Stop:** 30 s of silence. **Pre-roll:** 15 s.
- **Results** (4 h, 73 far-field/Indonesian conversations at 0–20 dB over MUSAN noise):
  - all conversations recorded;
  - first words kept in 95 % of conversations at ≥ 10 dB;
  - 0.25 false triggers/h outside music (sung vocals trigger, like TV).
- **Rejected:** earshot. 40 % of MUSAN noise files contain ≥ 1 s of its "speech", and it makes 9–15 false starts/h.

## Layout
| Path | What |
|---|---|
| `crates/core` | Pure decision logic: noise floor, per-stream detector, the recording `Controller` state machine. No I/O. |
| `crates/vad` | `VadEngine` trait with Silero v6.2 (pure Rust) and earshot. |
| `crates/audio` | DSP blocks: downmix, 90 Hz high-pass, RNNoise noise suppression, resampling (no allocation per call). |
| `tools/vad-eval` | Offline evaluation (`run`, `synth`, `mix`, `scan`, `trace`): onset latency, misses, false triggers, CPU. `run --frontend raw\|hp\|hp+ns` (default `hp+ns`, as the app does). |
| `tools/vad-live` | Live mic → VAD → controller, prints START/STOP. Records nothing. |
| `third_party/silero-vad-pure` | Vendored, reviewed and patched Silero port (see its `VENDORED.md`). |
| `docs/decisions` | Architecture decision records. |

## Develop (NixOS)
```sh
nix-shell                                   # pkg-config, alsa-lib, ffmpeg
scripts/fetch-golden.sh                     # test audio for the Silero parity test
cargo test --workspace --release
PROPTEST_CASES=20000 cargo test --release -p recorder-core --test properties
cargo clippy --workspace --all-targets -- -D warnings
```

## Measure detection
```sh
cargo build --release -p vad-eval
target/release/vad-eval synth --speech testdata/golden/aepyx.wav --out testdata/corpus/noisy.wav \
    --noise-dbfs -40 --speech-dbfs -30            # writes noisy.wav + noisy.wav.labels.csv
target/release/vad-eval run --engine both testdata/corpus/noisy.wav
target/release/vad-eval run --engine both --set energy_margin_db=3 testdata/corpus/noisy.wav
```
Labels: `<file>.labels.csv` with one `start_sec,end_sec` per conversation.

### Real corpus (open licenses, about 12.5 GB)
```sh
scripts/fetch-corpus.sh          # AMI far-field (24 meetings) + FLEURS-id + MUSAN, all CC BY 4.0
target/release/vad-eval mix --streams 8 --minutes 30     # -> testdata/corpus/real/mixNN_snrXX.wav
target/release/vad-eval run --engine both testdata/corpus/real/*.wav
```
Each stream has:
- a MUSAN noise bed;
- conversations at 20/10/5/0 dB SNR: AMI far-field excerpts, or Indonesian FLEURS "dialogues" with turn gaps and reverb;
- unlabeled music clips, so `run` can report false triggers inside music separately.

## Tune live
```sh
nix-shell --run "cargo run --release -p vad-live -- --list"
nix-shell --run "cargo run --release -p vad-live -- --engine earshot"
```
If it reports pure digital silence, the input is muted: `wpctl set-mute @DEFAULT_SOURCE@ 0`.

## Silero golden reference
`testdata/fixtures/silero_v6.2.3_ctx_aepyx_16k.json` is produced by
`scripts/gen-golden-official.py` with the official v6.2.3 model and onnxruntime, run
the way the official `OnnxWrapper` does it (64-sample context):
```sh
nix-shell -p "python3.withPackages (ps: [ ps.onnxruntime ps.numpy ])" --run "python3 scripts/gen-golden-official.py"
```
