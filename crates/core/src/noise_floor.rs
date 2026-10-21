//! Background level estimate: a percentile of recent *non-speech* frame levels.
//!
//! Only frames the VAD considers non-speech are fed in. Estimating from all
//! frames would let a long monologue raise the "floor" to speech level and
//! close the energy gate mid-sentence.

const MIN_DB: f32 = -100.0;
const MAX_DB: f32 = 0.0;
const BIN_DB: f32 = 0.5;
const BINS: usize = ((MAX_DB - MIN_DB) / BIN_DB) as usize + 1;

/// Sliding-window percentile over a fixed number of frames, via a histogram of
/// 0.5 dB bins. O(bins) per query, no allocation after construction.
#[derive(Debug, Clone)]
pub struct NoiseFloor {
    ring: Vec<u16>,
    head: usize,
    len: usize,
    hist: Vec<u32>,
    percentile: f32,
    /// Below this many samples the estimate is untrustworthy and the gate stays open.
    min_samples: usize,
}

impl NoiseFloor {
    pub fn new(window_frames: usize, percentile: f32, min_samples: usize) -> Self {
        let window_frames = window_frames.max(1);
        Self {
            ring: vec![0; window_frames],
            head: 0,
            len: 0,
            hist: vec![0; BINS],
            percentile,
            min_samples: min_samples.min(window_frames),
        }
    }

    fn bin(db: f32) -> u16 {
        let db = if db.is_finite() { db.clamp(MIN_DB, MAX_DB) } else { MIN_DB };
        ((db - MIN_DB) / BIN_DB).round() as u16
    }

    pub fn push(&mut self, db: f32) {
        let b = Self::bin(db);
        if self.len == self.ring.len() {
            let old = self.ring[self.head];
            self.hist[old as usize] -= 1;
        } else {
            self.len += 1;
        }
        self.ring[self.head] = b;
        self.hist[b as usize] += 1;
        self.head = (self.head + 1) % self.ring.len();
    }

    /// Estimated background level in dBFS, or `None` while still warming up.
    pub fn floor_db(&self) -> Option<f32> {
        if self.len < self.min_samples.max(1) {
            return None;
        }
        let target = ((self.len as f32) * self.percentile).floor() as u32;
        let mut seen = 0u32;
        for (i, &count) in self.hist.iter().enumerate() {
            seen += count;
            if seen > target {
                return Some(MIN_DB + i as f32 * BIN_DB);
            }
        }
        Some(MAX_DB)
    }

    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
        self.hist.iter_mut().for_each(|c| *c = 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warms_up_then_tracks_low_percentile() {
        let mut nf = NoiseFloor::new(100, 0.10, 10);
        for _ in 0..9 {
            nf.push(-60.0);
        }
        assert_eq!(nf.floor_db(), None);
        nf.push(-60.0);
        assert_eq!(nf.floor_db(), Some(-60.0));
        // 10th percentile ignores the occasional louder frame.
        for i in 0..90 {
            nf.push(if i % 3 == 0 { -30.0 } else { -55.0 });
        }
        let f = nf.floor_db().unwrap();
        assert!((-60.0..=-55.0).contains(&f), "{f}");
    }

    #[test]
    fn window_slides() {
        let mut nf = NoiseFloor::new(10, 0.10, 1);
        for _ in 0..10 {
            nf.push(-70.0);
        }
        for _ in 0..10 {
            nf.push(-40.0);
        }
        assert_eq!(nf.floor_db(), Some(-40.0));
    }

    #[test]
    fn clamps_garbage() {
        let mut nf = NoiseFloor::new(4, 0.0, 1);
        nf.push(f32::NAN);
        nf.push(f32::NEG_INFINITY);
        nf.push(12.0);
        assert_eq!(nf.floor_db(), Some(-100.0));
    }
}
