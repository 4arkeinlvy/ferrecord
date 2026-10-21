//! DSP building blocks for the capture path.
//!
//! Everything here allocates at construction only, so it can run on the
//! real-time DSP thread: [`downmix`] interleaved device audio to mono,
//! [`HighPass`] away HVAC rumble, [`NoiseSuppressor`] (RNNoise) at 48 kHz, and
//! [`To16k`] resample for the VAD.
//!
//! VAD front end, chosen by measurement on the real corpus (`vad-eval`):
//! high-pass -> noise suppression at 48 kHz -> 16 kHz. Noise suppression took
//! far-field coverage from 82 % to 95 % and left no conversation unrecorded.
#![forbid(unsafe_code)]

use std::f32::consts::PI;
use std::fmt;

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters};

/// Rate the VAD engines expect.
pub const VAD_RATE: u32 = 16_000;

/// Averages interleaved channels into `out` (cleared first; reuses its capacity).
pub fn downmix(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    out.clear();
    if channels <= 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    let scale = 1.0 / channels as f32;
    out.extend(interleaved.chunks_exact(channels).map(|f| f.iter().sum::<f32>() * scale));
}

/// Second-order Butterworth high-pass (RBJ cookbook biquad, Q = 1/sqrt 2).
#[derive(Debug, Clone)]
pub struct HighPass {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl HighPass {
    pub fn new(sample_rate: f32, cutoff_hz: f32) -> Self {
        let w0 = 2.0 * PI * cutoff_hz / sample_rate;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 + cos) / 2.0 / a0,
            b1: -(1.0 + cos) / a0,
            b2: (1.0 + cos) / 2.0 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Filters in place (transposed direct form II).
    pub fn process(&mut self, buf: &mut [f32]) {
        for x in buf {
            let y = self.b0 * *x + self.z1;
            self.z1 = self.b1 * *x - self.a1 * y + self.z2;
            self.z2 = self.b2 * *x - self.a2 * y;
            *x = y;
        }
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

#[derive(Debug)]
pub struct ResampleError(String);

impl fmt::Display for ResampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "resampler: {}", self.0)
    }
}

impl std::error::Error for ResampleError {}

/// Sample rate RNNoise operates at.
pub const NS_RATE: u32 = 48_000;

/// RNNoise noise suppression (pure-Rust `nnnoiseless`) for the VAD path.
///
/// Takes 48 kHz mono in `[-1, 1]` in blocks of any size and emits denoised
/// 10 ms frames. Adds 10 ms of latency (one frame buffered).
pub struct NoiseSuppressor {
    state: Box<nnnoiseless::DenoiseState<'static>>,
    input: Vec<f32>,
    filled: usize,
    output: Vec<f32>,
}

impl NoiseSuppressor {
    pub const FRAME: usize = nnnoiseless::DenoiseState::FRAME_SIZE;

    pub fn new() -> Self {
        Self {
            state: nnnoiseless::DenoiseState::new(),
            input: vec![0.0; Self::FRAME],
            filled: 0,
            output: vec![0.0; Self::FRAME],
        }
    }

    /// Feeds 48 kHz mono samples; `sink` receives each denoised 480-sample frame.
    pub fn push(&mut self, mut samples: &[f32], mut sink: impl FnMut(&[f32])) {
        while !samples.is_empty() {
            let take = (Self::FRAME - self.filled).min(samples.len());
            // RNNoise works on a 16-bit scale.
            for (d, s) in self.input[self.filled..self.filled + take].iter_mut().zip(&samples[..take]) {
                *d = s * 32768.0;
            }
            self.filled += take;
            samples = &samples[take..];
            if self.filled == Self::FRAME {
                self.filled = 0;
                self.state.process_frame(&mut self.output, &self.input);
                for v in self.output.iter_mut() {
                    *v /= 32768.0;
                }
                sink(&self.output);
            }
        }
    }

    pub fn reset(&mut self) {
        self.state = nnnoiseless::DenoiseState::new();
        self.filled = 0;
    }
}

impl Default for NoiseSuppressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Streams mono audio between two sample rates with a windowed-sinc filter
/// (no aliasing). Equal rates pass straight through.
pub struct StreamResampler {
    resampler: Option<Async<f32>>,
    input: Vec<f32>,
    filled: usize,
    output: Vec<f32>,
}

impl StreamResampler {
    /// Input frames consumed per resampler call (10 ms at 48 kHz).
    const CHUNK: usize = 480;

    pub fn new(from: u32, to: u32) -> Result<Self, ResampleError> {
        if from == 0 || to == 0 {
            return Err(ResampleError("sample rates must be positive".into()));
        }
        if from == to {
            return Ok(Self { resampler: None, input: Vec::new(), filled: 0, output: Vec::new() });
        }
        let ratio = f64::from(to) / f64::from(from);
        let params = SincInterpolationParameters::default();
        let resampler = Async::<f32>::new_sinc(ratio, 1.0, &params, Self::CHUNK, 1, FixedAsync::Input)
            .map_err(|e| ResampleError(e.to_string()))?;
        let output = vec![0.0; resampler.output_frames_max()];
        Ok(Self { resampler: Some(resampler), input: vec![0.0; Self::CHUNK], filled: 0, output })
    }

    /// Feeds mono samples; `sink` receives resampled blocks.
    pub fn push(&mut self, mut samples: &[f32], mut sink: impl FnMut(&[f32])) -> Result<(), ResampleError> {
        let Some(resampler) = self.resampler.as_mut() else {
            sink(samples);
            return Ok(());
        };
        while !samples.is_empty() {
            let take = (Self::CHUNK - self.filled).min(samples.len());
            self.input[self.filled..self.filled + take].copy_from_slice(&samples[..take]);
            self.filled += take;
            samples = &samples[take..];
            if self.filled < Self::CHUNK {
                break;
            }
            self.filled = 0;
            let cap = self.output.len();
            let input = InterleavedSlice::new(&self.input[..], 1, Self::CHUNK)
                .map_err(|e| ResampleError(e.to_string()))?;
            let mut output = InterleavedSlice::new_mut(&mut self.output[..], 1, cap)
                .map_err(|e| ResampleError(e.to_string()))?;
            let (_, written) = resampler
                .process_into_buffer(&input, &mut output, Some(&Indexing::new()))
                .map_err(|e| ResampleError(e.to_string()))?;
            sink(&self.output[..written]);
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.filled = 0;
        if let Some(r) = self.resampler.as_mut() {
            r.reset();
        }
    }
}

/// Streams mono audio at any device rate to 16 kHz for the VAD.
pub struct To16k(StreamResampler);

impl To16k {
    pub fn new(input_rate: u32) -> Result<Self, ResampleError> {
        StreamResampler::new(input_rate, VAD_RATE).map(Self)
    }

    /// Feeds mono samples; `sink` receives resampled 16 kHz blocks.
    pub fn push(&mut self, samples: &[f32], sink: impl FnMut(&[f32])) -> Result<(), ResampleError> {
        self.0.push(samples, sink)
    }

    pub fn reset(&mut self) {
        self.0.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, freq: f32, secs: f32) -> Vec<f32> {
        let n = (rate as f32 * secs) as usize;
        (0..n).map(|i| (2.0 * PI * freq * i as f32 / rate as f32).sin()).collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    fn zero_crossings(x: &[f32]) -> usize {
        x.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count()
    }

    #[test]
    fn downmix_averages_channels() {
        let mut out = Vec::new();
        downmix(&[1.0, 3.0, -2.0, 0.0], 2, &mut out);
        assert_eq!(out, vec![2.0, -1.0]);
        downmix(&[0.5, 0.25], 1, &mut out);
        assert_eq!(out, vec![0.5, 0.25]);
    }

    #[test]
    fn high_pass_removes_rumble_keeps_speech_band() {
        let rate = 48_000;
        let mut hp = HighPass::new(rate as f32, 90.0);
        let mut rumble = sine(rate, 25.0, 1.0);
        hp.process(&mut rumble);
        hp.reset();
        let mut voice = sine(rate, 1_000.0, 1.0);
        hp.process(&mut voice);
        let skip = rate as usize / 10; // ignore the filter's start-up transient
        assert!(rms(&rumble[skip..]) < 0.1, "25 Hz leaked: {}", rms(&rumble[skip..]));
        assert!((rms(&voice[skip..]) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.02);
    }

    fn resample_all(rate: u32, input: &[f32]) -> Vec<f32> {
        let mut r = To16k::new(rate).unwrap();
        let mut out = Vec::new();
        // Odd block sizes, like a real device callback.
        for block in input.chunks(333) {
            r.push(block, |b| out.extend_from_slice(b)).unwrap();
        }
        out
    }

    #[test]
    fn resamples_common_device_rates_to_16k() {
        for rate in [48_000, 44_100, 32_000] {
            let out = resample_all(rate, &sine(rate, 1_000.0, 2.0));
            let expected = 2.0 * VAD_RATE as f32;
            assert!((out.len() as f32 - expected).abs() < 0.02 * expected, "{rate}: {} samples", out.len());
            // 1 kHz tone keeps its pitch: ~2000 zero crossings per second.
            let tail = &out[out.len() / 4..];
            let per_sec = zero_crossings(tail) as f32 / (tail.len() as f32 / VAD_RATE as f32);
            assert!((per_sec - 2_000.0).abs() < 40.0, "{rate}: {per_sec} crossings/s");
        }
    }

    #[test]
    fn content_above_8khz_does_not_alias_into_the_speech_band() {
        let rate = 48_000;
        let out = resample_all(rate, &sine(rate, 12_000.0, 1.0));
        let tail = &out[out.len() / 4..];
        assert!(rms(tail) < 0.01, "12 kHz aliased with rms {}", rms(tail));
    }

    #[test]
    fn upsampling_round_trip_keeps_the_signal() {
        let input = sine(VAD_RATE, 700.0, 1.0);
        let mut up = StreamResampler::new(VAD_RATE, 48_000).unwrap();
        let mut down = StreamResampler::new(48_000, VAD_RATE).unwrap();
        let mut mid = Vec::new();
        for b in input.chunks(250) {
            up.push(b, |x| mid.extend_from_slice(x)).unwrap();
        }
        assert!((mid.len() as f32 / input.len() as f32 - 3.0).abs() < 0.05);
        let mut out = Vec::new();
        down.push(&mid, |x| out.extend_from_slice(x)).unwrap();
        let tail = &out[out.len() / 4..];
        let per_sec = zero_crossings(tail) as f32 / (tail.len() as f32 / VAD_RATE as f32);
        assert!((per_sec - 1_400.0).abs() < 30.0, "{per_sec}");
        assert!((rms(tail) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.03);
    }

    #[test]
    fn noise_suppressor_attenuates_steady_noise_and_keeps_length() {
        let mut rng_state = 0x2545_f491u32;
        let mut noise = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            (rng_state as f32 / u32::MAX as f32 - 0.5) * 0.05
        };
        let input: Vec<f32> = (0..NS_RATE as usize * 2).map(|_| noise()).collect();
        let mut ns = NoiseSuppressor::new();
        let mut out = Vec::new();
        for block in input.chunks(333) {
            ns.push(block, |f| out.extend_from_slice(f));
        }
        assert_eq!(out.len(), input.len() / NoiseSuppressor::FRAME * NoiseSuppressor::FRAME);
        let settle = NS_RATE as usize / 2;
        let ratio_db = 20.0 * (rms(&out[settle..]) / rms(&input[settle..out.len()])).log10();
        assert!(ratio_db < -6.0, "white noise only reduced by {ratio_db:.1} dB");
    }

    #[test]
    fn sixteen_k_passes_through() {
        let input = sine(VAD_RATE, 440.0, 0.1);
        assert_eq!(resample_all(VAD_RATE, &input), input);
    }
}
