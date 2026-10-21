//! The `SileroVad` state machine: STFT -> encoder -> LSTM -> head.

use std::path::Path;

use crate::conv::conv1d_relu_auto;
use crate::lstm::{HIDDEN, LstmState};
use crate::stft::{StftScratch, stft_mag};
use crate::weights::{RateWeights, Weights};
use crate::{Error, SampleRate};

/// Per-sample-rate pipeline constants, derived from the exported STFT basis
/// shape: kernel K, hop K/2, right reflect pad K/4, chunk 2K.
struct Config {
    chunk: usize,
    /// Samples of the previous input prepended to each chunk (official
    /// `OnnxWrapper`: 64 at 16 kHz, 32 at 8 kHz). Patched in when vendored.
    context: usize,
    pad: usize,
    n_bins: usize,
    kernel: usize,
}

impl Config {
    fn for_rate(sample_rate: SampleRate) -> Self {
        let (chunk, context, pad, n_bins, kernel) = match sample_rate {
            SampleRate::Hz16000 => (512, 64, 64, 129, 256),
            SampleRate::Hz8000 => (256, 32, 32, 65, 128),
        };
        Self { chunk, context, pad, n_bins, kernel }
    }
}

/// Per-layer outputs of a single `process` call, for golden-data comparison
/// and debugging. All matrices are [channels, time] row-major.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct Trace {
    pub stft_mag: Vec<f32>,
    pub stft_t: usize,
    pub enc: [Vec<f32>; 4],
    pub head_logits: Vec<f32>,
    pub h: [f32; HIDDEN],
    pub c: [f32; HIDDEN],
}

impl Trace {
    fn empty() -> Self {
        Self {
            stft_mag: Vec::new(),
            stft_t: 0,
            enc: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            head_logits: Vec::new(),
            h: [0.0; HIDDEN],
            c: [0.0; HIDDEN],
        }
    }
}

/// Reusable hot-path buffers, allocated once per `SileroVad`.
struct Scratch {
    stft: StftScratch,
    /// STFT magnitudes, [n_bins, t_stft].
    mag: Vec<f32>,
    /// im2col scratch for the encoder fallback path.
    im2col: Vec<f32>,
    /// Padded-input scratch for the NEON direct-conv kernel.
    #[cfg(target_arch = "aarch64")]
    conv_xpad: Vec<f32>,
    /// Encoder ping-pong activation buffers.
    enc_a: Vec<f32>,
    enc_b: Vec<f32>,
}

impl Scratch {
    fn new(cfg: &Config) -> Self {
        // Model input is context + chunk; (input + pad - kernel) / (kernel/2) + 1.
        let input = cfg.context + cfg.chunk;
        let t_stft = (input + cfg.pad - cfg.kernel) / (cfg.kernel / 2) + 1;
        let enc0_out = 128 * t_stft;
        Self {
            stft: StftScratch { xp: Vec::with_capacity(input + cfg.pad), ..Default::default() },
            mag: Vec::with_capacity(cfg.n_bins * t_stft),
            im2col: Vec::with_capacity(cfg.n_bins * 3 * t_stft),
            #[cfg(target_arch = "aarch64")]
            conv_xpad: Vec::with_capacity(256 * 8),
            enc_a: Vec::with_capacity(enc0_out),
            enc_b: Vec::with_capacity(enc0_out),
        }
    }
}

/// Streaming silero-vad inference. Feed fixed-size chunks of f32 samples
/// (normalized to [-1, 1]); get a speech probability per chunk.
pub struct SileroVad {
    weights: RateWeights,
    config: Config,
    lstm: LstmState,
    scratch: Scratch,
    /// Last `config.context` samples of the previous model input.
    context: Vec<f32>,
    /// Preallocated `context ++ chunk` model input.
    input: Vec<f32>,
    #[cfg(feature = "profile")]
    profile: StageProfile,
}

/// Accumulated per-stage wall time across `process` calls. Only available
/// with the `profile` cargo feature; when the feature is off there is no
/// instrumentation in the hot path at all.
#[cfg(feature = "profile")]
#[derive(Debug, Default, Clone, Copy)]
pub struct StageProfile {
    /// STFT magnitude.
    pub stft: std::time::Duration,
    /// Encoder conv layers 0..4.
    pub enc: [std::time::Duration; 4],
    /// LSTM steps + head (incl. final sigmoid).
    pub lstm_head: std::time::Duration,
    /// Number of chunks processed.
    pub chunks: u64,
}

impl SileroVad {
    /// Create with the embedded weights (`assets/silero_vad.bin`).
    pub fn new(sample_rate: SampleRate) -> Result<Self, Error> {
        let mut w = Weights::embedded()?;
        Ok(Self::build(RateWeights::select(&mut w, sample_rate)?, sample_rate))
    }

    /// Create with weights loaded from an external `.bin` file.
    pub fn with_weights(sample_rate: SampleRate, path: &Path) -> Result<Self, Error> {
        let mut w = Weights::from_file(path)?;
        Ok(Self::build(RateWeights::select(&mut w, sample_rate)?, sample_rate))
    }

    /// Clear the LSTM state and the input context (start of a new audio stream).
    pub fn reset(&mut self) {
        self.lstm.reset();
        self.context.fill(0.0);
    }

    fn build(weights: RateWeights, sample_rate: SampleRate) -> Self {
        let config = Config::for_rate(sample_rate);
        Self {
            weights,
            lstm: LstmState::new(),
            scratch: Scratch::new(&config),
            context: vec![0.0; config.context],
            input: Vec::with_capacity(config.context + config.chunk),
            config,
            #[cfg(feature = "profile")]
            profile: StageProfile::default(),
        }
    }

    /// Run the model on `context ++ chunk`, then keep the chunk's tail as the
    /// next context (exactly what the official `OnnxWrapper.__call__` does).
    fn run_with_context(&mut self, chunk: &[f32], trace: Option<&mut Trace>) -> Result<f32, Error> {
        if chunk.len() != self.config.chunk {
            return Err(Error::BadChunkSize { expected: self.config.chunk, got: chunk.len() });
        }
        let mut input = std::mem::take(&mut self.input);
        input.clear();
        input.extend_from_slice(&self.context);
        input.extend_from_slice(chunk);
        let result = self.forward(&input, trace);
        let tail = input.len() - self.config.context;
        self.context.copy_from_slice(&input[tail..]);
        self.input = input;
        result
    }

    /// Required chunk length in samples: 512 at 16kHz, 256 at 8kHz.
    pub fn chunk_size(&self) -> usize {
        self.config.chunk
    }

    /// Process one chunk, returning the speech probability in [0, 1].
    ///
    /// The probability is the mean of the per-time-step sigmoid outputs of
    /// the head (ONNX `ReduceMean` over the time axis — with T' = 1 step per
    /// chunk in this model it equals that step's output).
    pub fn process(&mut self, chunk: &[f32]) -> Result<f32, Error> {
        self.run_with_context(chunk, None)
    }

    /// Like `process`, but also returns per-layer intermediates.
    #[doc(hidden)]
    pub fn process_traced(&mut self, chunk: &[f32]) -> Result<(f32, Trace), Error> {
        let mut trace = Trace::empty();
        let prob = self.run_with_context(chunk, Some(&mut trace))?;
        Ok((prob, trace))
    }

    /// Current LSTM state (for golden comparison / checkpointing).
    #[doc(hidden)]
    pub fn state(&self) -> ([f32; HIDDEN], [f32; HIDDEN]) {
        (self.lstm.h, self.lstm.c)
    }

    /// Accumulated per-stage timings (only with the `profile` feature).
    #[cfg(feature = "profile")]
    pub fn stage_profile(&self) -> StageProfile {
        self.profile
    }

    /// Reset the accumulated per-stage timings.
    #[cfg(feature = "profile")]
    pub fn reset_profile(&mut self) {
        self.profile = StageProfile::default();
    }

    /// Model forward pass over `input` (= context ++ chunk).
    fn forward(&mut self, chunk: &[f32], mut trace: Option<&mut Trace>) -> Result<f32, Error> {
        debug_assert_eq!(chunk.len(), self.config.context + self.config.chunk);
        let cfg = &self.config;
        let w = &self.weights;
        let s = &mut self.scratch;
        #[cfg(target_arch = "aarch64")]
        let conv_xpad = &mut s.conv_xpad;
        let Scratch { stft, mag, im2col, enc_a, enc_b, .. } = s;

        // 1. STFT magnitude: [n_bins, T]
        #[cfg(feature = "profile")]
        let t_stft0 = std::time::Instant::now();
        let t = stft_mag(chunk, cfg.pad, &w.stft_basis.data, cfg.n_bins, cfg.kernel, stft, mag);
        #[cfg(feature = "profile")]
        {
            self.profile.stft += t_stft0.elapsed();
            self.profile.chunks += 1;
        }
        if let Some(tr) = trace.as_deref_mut() {
            tr.stft_mag.clone_from(mag);
            tr.stft_t = t;
        }

        // 2. Encoder: conv(k=3, pad=1) + ReLU, strides 1, 2, 2, 1.
        // Ping-pong between enc_a/enc_b; after 4 layers the result sits in
        // enc_b.
        let strides = [1usize, 2, 2, 1];
        let mut cin = cfg.n_bins;
        let mut t_cur = t;
        for (i, stride) in strides.iter().enumerate() {
            let (src, dst): (&[f32], &mut Vec<f32>) = match (i, i % 2) {
                (0, _) => (mag, enc_a),
                (_, 0) => (enc_b, enc_a),
                _ => (enc_a, enc_b),
            };
            let cout = w.enc_w[i].dims[0];
            #[cfg(feature = "profile")]
            let t_enc0 = std::time::Instant::now();
            t_cur = conv1d_relu_auto(
                src,
                cin,
                t_cur,
                &w.enc_w[i].data,
                #[cfg(target_arch = "aarch64")]
                &w.enc_w_packed[i],
                &w.enc_b[i],
                cout,
                3,
                1,
                *stride,
                #[cfg(target_arch = "aarch64")]
                conv_xpad,
                im2col,
                dst,
            );
            #[cfg(feature = "profile")]
            {
                self.profile.enc[i] += t_enc0.elapsed();
            }
            if let Some(tr) = trace.as_deref_mut() {
                tr.enc[i].clone_from(dst);
            }
            cin = cout;
        }
        let t_enc = t_cur;
        debug_assert_eq!(cin, HIDDEN);
        let enc_out = enc_b; // 4 layers, last write went to enc_b

        // 3. LSTM over time steps; 4. head per step: relu -> 1x1 conv -> sigmoid
        #[cfg(feature = "profile")]
        let t_lstm0 = std::time::Instant::now();
        let mut prob_sum = 0.0f32;
        for step in 0..t_enc {
            let mut x = [0.0f32; HIDDEN];
            for (j, xj) in x.iter_mut().enumerate() {
                *xj = enc_out[j * t_enc + step];
            }
            self.lstm.step(&w.lstm_w.data, &w.lstm_b, &x);
            let logit: f32 =
                self.lstm.h.iter().zip(&w.head_w).map(|(h, wj)| h.max(0.0) * wj).sum::<f32>() + w.head_b;
            if let Some(tr) = trace.as_deref_mut() {
                tr.head_logits.push(logit);
            }
            prob_sum += 1.0 / (1.0 + (-logit).exp());
        }
        let prob = prob_sum / t_enc as f32;
        #[cfg(feature = "profile")]
        {
            self.profile.lstm_head += t_lstm0.elapsed();
        }

        if let Some(tr) = trace {
            tr.h = self.lstm.h;
            tr.c = self.lstm.c;
        }
        Ok(prob)
    }
}
