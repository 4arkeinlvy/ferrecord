//! Voice activity detection engines behind one trait.
//!
//! Every engine consumes 16 kHz mono `f32` frames in `[-1, 1]` of a fixed,
//! engine-specific length and returns a speech probability in `[0, 1]`.
//! Engines are allocation-free per frame once constructed, so they can run on
//! the real-time DSP thread.
#![forbid(unsafe_code)]

use std::fmt;
use std::time::Duration;

/// Sample rate every engine expects.
pub const SAMPLE_RATE: u32 = 16_000;

/// Which engine to build. Silero is the default; earshot is the challenger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    Silero,
    Earshot,
}

impl EngineKind {
    pub const ALL: [EngineKind; 2] = [EngineKind::Silero, EngineKind::Earshot];

    pub fn name(self) -> &'static str {
        match self {
            EngineKind::Silero => "silero-v6.2",
            EngineKind::Earshot => "earshot",
        }
    }
}

impl std::str::FromStr for EngineKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "silero" | "silero-v6.2" => Ok(EngineKind::Silero),
            "earshot" => Ok(EngineKind::Earshot),
            other => Err(format!("unknown VAD engine `{other}` (expected silero or earshot)")),
        }
    }
}

#[derive(Debug)]
pub enum VadError {
    BadFrameLen { expected: usize, got: usize },
    Engine(String),
}

impl fmt::Display for VadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VadError::BadFrameLen { expected, got } => {
                write!(f, "VAD frame must be {expected} samples, got {got}")
            }
            VadError::Engine(msg) => write!(f, "VAD engine error: {msg}"),
        }
    }
}

impl std::error::Error for VadError {}

/// A voice activity detector for one audio stream.
pub trait VadEngine: Send {
    fn kind(&self) -> EngineKind;
    /// Samples per frame at [`SAMPLE_RATE`].
    fn frame_len(&self) -> usize;
    /// Duration of one frame.
    fn frame_duration(&self) -> Duration {
        Duration::from_micros(self.frame_len() as u64 * 1_000_000 / u64::from(SAMPLE_RATE))
    }
    /// Speech probability for exactly [`Self::frame_len`] samples.
    fn process(&mut self, frame: &[f32]) -> Result<f32, VadError>;
    /// Forget all recurrent state (new stream, device switch, wake from sleep).
    fn reset(&mut self);
}

/// Build an engine. Construction allocates; `process` does not.
pub fn new_engine(kind: EngineKind) -> Result<Box<dyn VadEngine>, VadError> {
    Ok(match kind {
        EngineKind::Silero => Box::new(Silero::new()?),
        EngineKind::Earshot => Box::new(Earshot::new()),
    })
}

/// Silero VAD v6.2 through the vendored pure-Rust forward pass (512 samples, 32 ms).
pub struct Silero {
    inner: silero_vad_pure::SileroVad,
}

impl Silero {
    pub fn new() -> Result<Self, VadError> {
        let inner = silero_vad_pure::SileroVad::new(silero_vad_pure::SampleRate::Hz16000)
            .map_err(|e| VadError::Engine(e.to_string()))?;
        Ok(Self { inner })
    }
}

impl VadEngine for Silero {
    fn kind(&self) -> EngineKind {
        EngineKind::Silero
    }

    fn frame_len(&self) -> usize {
        self.inner.chunk_size()
    }

    fn process(&mut self, frame: &[f32]) -> Result<f32, VadError> {
        self.inner.process(frame).map_err(|e| match e {
            silero_vad_pure::Error::BadChunkSize { expected, got } => VadError::BadFrameLen { expected, got },
            other => VadError::Engine(other.to_string()),
        })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

const EARSHOT_FRAME: usize = 256;

/// earshot (pure Rust, ~95 KiB): 256 samples, 16 ms.
pub struct Earshot {
    inner: Box<earshot::Detector>,
    clamped: [f32; EARSHOT_FRAME],
}

impl Earshot {
    pub fn new() -> Self {
        Self { inner: earshot::Detector::default_boxed(), clamped: [0.0; EARSHOT_FRAME] }
    }
}

impl Default for Earshot {
    fn default() -> Self {
        Self::new()
    }
}

impl VadEngine for Earshot {
    fn kind(&self) -> EngineKind {
        EngineKind::Earshot
    }

    fn frame_len(&self) -> usize {
        EARSHOT_FRAME
    }

    fn process(&mut self, frame: &[f32]) -> Result<f32, VadError> {
        if frame.len() != EARSHOT_FRAME {
            return Err(VadError::BadFrameLen { expected: EARSHOT_FRAME, got: frame.len() });
        }
        // earshot requires [-1, 1]; filters/AEC upstream can overshoot slightly.
        for (dst, src) in self.clamped.iter_mut().zip(frame) {
            *dst = src.clamp(-1.0, 1.0);
        }
        Ok(self.inner.predict_f32(&self.clamped).clamp(0.0, 1.0))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Frame level in dBFS (full scale = 1.0), floored at -100 dB.
pub fn rms_dbfs(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return -100.0;
    }
    let mean_sq = frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32;
    if mean_sq <= 1e-10 { -100.0 } else { (10.0 * mean_sq.log10()).max(-100.0) }
}

/// Re-chunks arbitrary-length sample blocks into fixed-size frames without
/// allocating after construction.
pub struct Framer {
    buf: Vec<f32>,
    filled: usize,
}

impl Framer {
    pub fn new(frame_len: usize) -> Self {
        Self { buf: vec![0.0; frame_len], filled: 0 }
    }

    pub fn frame_len(&self) -> usize {
        self.buf.len()
    }

    /// Feed samples; `on_frame` is called once per completed frame.
    pub fn push(&mut self, mut samples: &[f32], mut on_frame: impl FnMut(&[f32])) {
        while !samples.is_empty() {
            let take = (self.buf.len() - self.filled).min(samples.len());
            self.buf[self.filled..self.filled + take].copy_from_slice(&samples[..take]);
            self.filled += take;
            samples = &samples[take..];
            if self.filled == self.buf.len() {
                on_frame(&self.buf);
                self.filled = 0;
            }
        }
    }

    pub fn clear(&mut self) {
        self.filled = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, freq: f32, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin())
            .collect()
    }

    #[test]
    fn frame_lengths_and_durations() {
        let s = Silero::new().unwrap();
        assert_eq!(s.frame_len(), 512);
        assert_eq!(s.frame_duration(), Duration::from_millis(32));
        let e = Earshot::new();
        assert_eq!(e.frame_len(), 256);
        assert_eq!(e.frame_duration(), Duration::from_millis(16));
    }

    #[test]
    fn engines_reject_wrong_frame_len() {
        for kind in EngineKind::ALL {
            let mut eng = new_engine(kind).unwrap();
            let err = eng.process(&[0.0; 100]).unwrap_err();
            assert!(matches!(err, VadError::BadFrameLen { got: 100, .. }), "{kind:?}: {err}");
        }
    }

    #[test]
    fn silence_scores_low() {
        for kind in EngineKind::ALL {
            let mut eng = new_engine(kind).unwrap();
            let frame = vec![0.0; eng.frame_len()];
            let mut last = 1.0;
            for _ in 0..40 {
                last = eng.process(&frame).unwrap();
                assert!((0.0..=1.0).contains(&last));
            }
            assert!(last < 0.3, "{kind:?} scored digital silence at {last}");
        }
    }

    #[test]
    fn silero_context_carries_across_chunks_and_reset_clears_it() {
        let speechy: Vec<f32> =
            (0..512).map(|i| 0.3 * (i as f32 * 0.07).sin() + 0.2 * (i as f32 * 0.31).sin()).collect();
        let quiet = vec![0.0f32; 512];

        // The same chunk after different predecessors gives different results...
        let mut a = Silero::new().unwrap();
        let mut b = Silero::new().unwrap();
        a.process(&speechy).unwrap();
        b.process(&quiet).unwrap();
        let (pa, pb) = (a.process(&quiet).unwrap(), b.process(&quiet).unwrap());
        assert_ne!(pa, pb);

        // ...and reset() makes an engine indistinguishable from a fresh one.
        a.reset();
        let mut fresh = Silero::new().unwrap();
        for chunk in [&speechy, &quiet, &speechy] {
            assert_eq!(a.process(chunk).unwrap(), fresh.process(chunk).unwrap());
        }
    }

    #[test]
    fn earshot_clamps_out_of_range_input() {
        let mut e = Earshot::new();
        let loud = vec![1.7; 256];
        let p = e.process(&loud).unwrap();
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn rms_levels() {
        assert_eq!(rms_dbfs(&[]), -100.0);
        assert_eq!(rms_dbfs(&[0.0; 64]), -100.0);
        // Full-scale sine has RMS 1/sqrt(2) = -3.01 dBFS.
        let db = rms_dbfs(&tone(1600, 440.0, 1.0));
        assert!((db + 3.01).abs() < 0.05, "{db}");
    }

    #[test]
    fn framer_rechunks_without_loss() {
        let mut f = Framer::new(4);
        let mut frames = Vec::new();
        f.push(&[1.0, 2.0, 3.0], |fr| frames.push(fr.to_vec()));
        f.push(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0], |fr| frames.push(fr.to_vec()));
        assert_eq!(frames, vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]);
        f.push(&[10.0, 11.0, 12.0], |fr| frames.push(fr.to_vec()));
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[2], vec![9.0, 10.0, 11.0, 12.0]);
    }
}
