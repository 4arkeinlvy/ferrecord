//! Per-stream speech detection on top of raw VAD probabilities.
//!
//! A frame is *voiced* when the VAD probability passes a hysteresis threshold
//! and the frame is loud enough relative to the background. Voiced frames are
//! merged into segments (short gaps bridged); a segment that lasts long enough
//! produces keep-alives, so coughs and door clicks never hold a recording open.

use std::time::Duration;

use crate::config::Config;
use crate::noise_floor::NoiseFloor;

/// One analysed VAD frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameObs {
    /// Start of the frame on the shared media timeline.
    pub start: Duration,
    pub len: Duration,
    /// VAD speech probability in `[0, 1]`.
    pub prob: f32,
    /// Frame level in dBFS.
    pub rms_db: f32,
}

impl FrameObs {
    pub fn end(&self) -> Duration {
        self.start + self.len
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameVerdict {
    /// The frame is speech (hysteresis + energy gate).
    pub voiced: bool,
    /// The frame belongs to a segment long enough to count as activity.
    pub keepalive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamParams {
    pub enter: f32,
    pub exit: f32,
    pub energy_margin_db: f32,
    pub merge_gap: Duration,
    pub min_blip: Duration,
    pub keepalive: Duration,
}

impl StreamParams {
    pub fn mic(cfg: &Config) -> Self {
        Self::with_keepalive(cfg, cfg.keepalive_mic)
    }

    pub fn loopback(cfg: &Config) -> Self {
        Self::with_keepalive(cfg, cfg.keepalive_loopback)
    }

    fn with_keepalive(cfg: &Config, keepalive: Duration) -> Self {
        Self {
            enter: cfg.speech_enter,
            exit: cfg.speech_exit,
            energy_margin_db: cfg.energy_margin_db,
            merge_gap: cfg.merge_gap,
            min_blip: cfg.min_blip,
            keepalive: keepalive.max(cfg.min_blip),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Segment {
    start: Duration,
    last_voiced_end: Duration,
}

#[derive(Debug, Clone)]
pub struct StreamDetector {
    params: StreamParams,
    noise: NoiseFloor,
    hysteresis_on: bool,
    segment: Option<Segment>,
}

impl StreamDetector {
    pub fn new(params: StreamParams, frame_len: Duration, cfg: &Config) -> Self {
        let frame_us = frame_len.as_micros().max(1);
        let window_frames = cfg.noise_window.as_micros().div_ceil(frame_us) as usize;
        // Trust the floor after ~1 s of non-speech.
        let min_samples = 1_000_000u128.div_ceil(frame_us) as usize;
        Self {
            params,
            noise: NoiseFloor::new(window_frames, cfg.noise_percentile, min_samples),
            hysteresis_on: false,
            segment: None,
        }
    }

    pub fn observe(&mut self, f: &FrameObs) -> FrameVerdict {
        let p = &self.params;
        if self.hysteresis_on {
            if f.prob < p.exit {
                self.hysteresis_on = false;
            }
        } else if f.prob >= p.enter {
            self.hysteresis_on = true;
        }

        let loud_enough = match self.noise.floor_db() {
            Some(floor) => f.rms_db >= floor + p.energy_margin_db,
            None => true,
        };
        if f.prob < p.exit {
            self.noise.push(f.rms_db);
        }

        let voiced = self.hysteresis_on && loud_enough;
        let end = f.end();
        if voiced {
            match &mut self.segment {
                Some(seg) if f.start.saturating_sub(seg.last_voiced_end) <= p.merge_gap => {
                    seg.last_voiced_end = end;
                }
                _ => self.segment = Some(Segment { start: f.start, last_voiced_end: end }),
            }
        } else if let Some(seg) = self.segment
            && end.saturating_sub(seg.last_voiced_end) > p.merge_gap
        {
            self.segment = None;
        }

        let keepalive =
            voiced && self.segment.is_some_and(|s| s.last_voiced_end.saturating_sub(s.start) >= p.keepalive);
        FrameVerdict { voiced, keepalive }
    }

    pub fn noise_floor_db(&self) -> Option<f32> {
        self.noise.floor_db()
    }

    /// Forget segment/hysteresis state but keep the background estimate.
    pub fn reset_activity(&mut self) {
        self.hysteresis_on = false;
        self.segment = None;
    }

    /// Forget everything (device switch, wake from sleep).
    pub fn reset(&mut self) {
        self.reset_activity();
        self.noise.clear();
    }
}

/// Voiced frames inside a sliding time window, for the START rule.
///
/// Evicts by timestamp, not by frame count, so a gap in the frame stream can
/// never stretch the window beyond its configured duration.
#[derive(Debug, Clone)]
pub struct VoicedWindow {
    ends: Vec<Duration>,
    head: usize,
    len: usize,
    window: Duration,
    frame_len: Duration,
}

impl VoicedWindow {
    pub fn new(window: Duration, frame_len: Duration) -> Self {
        let cap = (window.as_micros().div_ceil(frame_len.as_micros().max(1)) as usize) + 1;
        Self { ends: vec![Duration::ZERO; cap], head: 0, len: 0, window, frame_len }
    }

    fn oldest(&self) -> Duration {
        self.ends[(self.head + self.ends.len() - self.len) % self.ends.len()]
    }

    /// Record a frame ending at `end`.
    pub fn push(&mut self, voiced: bool, end: Duration) {
        let horizon = end.saturating_sub(self.window);
        while self.len > 0 && self.oldest() <= horizon {
            self.len -= 1;
        }
        if voiced {
            self.ends[self.head] = end;
            self.head = (self.head + 1) % self.ends.len();
            self.len = (self.len + 1).min(self.ends.len());
        }
    }

    /// Speech inside the window as of the last pushed frame.
    pub fn voiced_duration(&self) -> Duration {
        self.frame_len * self.len as u32
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: Duration = Duration::from_millis(32);

    fn obs(i: u64, prob: f32, rms_db: f32) -> FrameObs {
        FrameObs { start: FRAME * i as u32, len: FRAME, prob, rms_db }
    }

    fn detector() -> StreamDetector {
        let cfg = Config::default();
        StreamDetector::new(StreamParams::mic(&cfg), FRAME, &cfg)
    }

    /// Feed `n` frames starting at index `from`; returns the verdicts.
    fn feed(d: &mut StreamDetector, from: u64, n: u64, prob: f32, db: f32) -> Vec<FrameVerdict> {
        (from..from + n).map(|i| d.observe(&obs(i, prob, db))).collect()
    }

    #[test]
    fn hysteresis_keeps_speech_through_mid_probabilities() {
        let mut d = detector();
        feed(&mut d, 0, 40, 0.05, -60.0); // warm the noise floor
        assert!(d.observe(&obs(40, 0.6, -25.0)).voiced);
        // 0.4 is below enter but above exit: stays voiced.
        assert!(d.observe(&obs(41, 0.4, -25.0)).voiced);
        assert!(!d.observe(&obs(42, 0.3, -25.0)).voiced);
        // 0.4 does not re-enter.
        assert!(!d.observe(&obs(43, 0.4, -25.0)).voiced);
    }

    #[test]
    fn energy_gate_rejects_speechlike_frames_near_background() {
        let mut d = detector();
        feed(&mut d, 0, 60, 0.05, -40.0);
        assert_eq!(d.noise_floor_db(), Some(-40.0));
        // VAD says speech but the level is only 3 dB over background.
        assert!(!d.observe(&obs(60, 0.9, -37.0)).voiced);
        assert!(d.observe(&obs(61, 0.9, -30.0)).voiced);
    }

    #[test]
    fn monologue_does_not_raise_the_floor() {
        let mut d = detector();
        feed(&mut d, 0, 60, 0.05, -60.0);
        // 60 s of continuous loud speech: the floor must not climb to speech level.
        let v = feed(&mut d, 60, 1875, 0.95, -25.0);
        assert!(v.iter().all(|v| v.voiced));
        assert_eq!(d.noise_floor_db(), Some(-60.0));
    }

    #[test]
    fn keepalive_needs_a_long_enough_segment() {
        let mut d = detector();
        feed(&mut d, 0, 40, 0.05, -60.0);
        // 7 frames = 224 ms: a blip, never a keep-alive.
        assert!(feed(&mut d, 40, 7, 0.9, -25.0).iter().all(|v| v.voiced && !v.keepalive));
        feed(&mut d, 47, 20, 0.05, -60.0);
        // 300 ms = 9.4 frames: the 10th frame of a run is the first keep-alive.
        let v = feed(&mut d, 67, 12, 0.9, -25.0);
        assert!(v[..9].iter().all(|v| !v.keepalive));
        assert!(v[9..].iter().all(|v| v.keepalive));
    }

    #[test]
    fn short_gaps_merge_into_one_segment() {
        let mut d = detector();
        feed(&mut d, 0, 40, 0.05, -60.0);
        feed(&mut d, 40, 6, 0.9, -25.0); // 192 ms
        feed(&mut d, 46, 5, 0.05, -60.0); // 160 ms gap (< 300 ms merge gap)
        let v = feed(&mut d, 51, 4, 0.9, -25.0);
        // Segment now spans 40..55 = 480 ms, so it qualifies.
        assert!(v.last().unwrap().keepalive);
    }

    #[test]
    fn long_gaps_split_segments() {
        let mut d = detector();
        feed(&mut d, 0, 40, 0.05, -60.0);
        feed(&mut d, 40, 6, 0.9, -25.0);
        feed(&mut d, 46, 12, 0.05, -60.0); // 384 ms gap
        let v = feed(&mut d, 58, 4, 0.9, -25.0);
        assert!(v.iter().all(|v| !v.keepalive));
    }

    #[test]
    fn voiced_window_counts_only_recent_frames() {
        let mut w = VoicedWindow::new(Duration::from_millis(160), FRAME); // 5 frames
        for (i, v) in [true, true, true, false, false].into_iter().enumerate() {
            w.push(v, FRAME * (i as u32 + 1));
        }
        assert_eq!(w.voiced_duration(), FRAME * 3);
        w.push(false, FRAME * 6);
        w.push(false, FRAME * 7);
        assert_eq!(w.voiced_duration(), FRAME);
        w.clear();
        assert_eq!(w.voiced_duration(), Duration::ZERO);
    }

    #[test]
    fn voiced_window_evicts_by_time_across_gaps() {
        let mut w = VoicedWindow::new(Duration::from_secs(2), FRAME);
        for i in 1..=20 {
            w.push(true, FRAME * i);
        }
        // Frames stop for 5 s; old speech must not count toward a new onset.
        w.push(true, Duration::from_secs(6));
        assert_eq!(w.voiced_duration(), FRAME);
    }
}
