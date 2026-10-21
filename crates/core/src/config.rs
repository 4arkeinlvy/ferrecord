//! Detector and recording-policy parameters.

use std::fmt;
use std::time::Duration;

/// All tunables. Defaults are the plan's starting values; M0 evaluation and the
/// pilot retune them, and IT policy can override them per site.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Frame counts as speech once the VAD probability reaches this...
    pub speech_enter: f32,
    /// ...and stays speech until it drops below this (hysteresis).
    pub speech_exit: f32,
    /// Speech must also be this far above the background level.
    pub energy_margin_db: f32,
    /// Window over which the background level is estimated.
    pub noise_window: Duration,
    /// Percentile of non-speech frame levels used as the background level.
    pub noise_percentile: f32,
    /// START needs `onset_min_speech` of speech inside the last `onset_window`.
    pub onset_window: Duration,
    pub onset_min_speech: Duration,
    /// Unvoiced gaps shorter than this don't split a speech segment.
    pub merge_gap: Duration,
    /// Segments shorter than this (coughs, doors) never count as speech activity.
    pub min_blip: Duration,
    /// A segment this long resets the silence timer (mic / system audio).
    pub keepalive_mic: Duration,
    pub keepalive_loopback: Duration,
    /// STOP after this long without a keep-alive.
    pub silence_stop: Duration,
    /// STOP if the mic itself has been silent this long (music on the PC can't
    /// hold a recording open through system audio).
    pub mic_silence_cap: Duration,
    /// Roll over to a new file after this much media.
    pub max_file: Duration,
    /// Files start this long before the trigger.
    pub preroll: Duration,
    /// Recordings with less mic speech than this are flagged as likely false triggers.
    pub likely_false_trigger_speech: Duration,
    /// After a manual stop, don't re-trigger until the mic has been quiet this long.
    pub rearm_silence: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            speech_enter: 0.50,
            speech_exit: 0.35,
            energy_margin_db: 6.0,
            noise_window: Duration::from_secs(30),
            noise_percentile: 0.10,
            onset_window: Duration::from_secs(2),
            onset_min_speech: Duration::from_secs(1),
            merge_gap: Duration::from_millis(300),
            min_blip: Duration::from_millis(250),
            keepalive_mic: Duration::from_millis(300),
            keepalive_loopback: Duration::from_millis(500),
            silence_stop: Duration::from_secs(30),
            mic_silence_cap: Duration::from_secs(5 * 60),
            max_file: Duration::from_secs(60 * 60),
            // Real-corpus eval (vad-eval): far-field triggers can come late, and 15 s
            // halves first-words-lost vs 8 s (17 -> 9 of 73). 20 s gets 5 but keeps
            // more pre-conversation audio, so it is left to policy.
            preroll: Duration::from_secs(15),
            likely_false_trigger_speech: Duration::from_secs(3),
            rearm_silence: Duration::from_secs(10),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid recorder config: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let err = |m: &str| Err(ConfigError(m.to_owned()));
        if !(0.0..=1.0).contains(&self.speech_enter) || !(0.0..=1.0).contains(&self.speech_exit) {
            return err("speech thresholds must be within [0, 1]");
        }
        if self.speech_exit > self.speech_enter {
            return err("speech_exit must not exceed speech_enter");
        }
        if !(0.0..1.0).contains(&self.noise_percentile) {
            return err("noise_percentile must be within [0, 1)");
        }
        if self.onset_window.is_zero() || self.onset_min_speech > self.onset_window {
            return err("onset_min_speech must fit inside a non-empty onset_window");
        }
        if self.onset_min_speech.is_zero() {
            return err("onset_min_speech must be positive");
        }
        if self.silence_stop.is_zero() || self.max_file.is_zero() || self.noise_window.is_zero() {
            return err("silence_stop, max_file and noise_window must be positive");
        }
        if self.mic_silence_cap < self.silence_stop {
            return err("mic_silence_cap must be at least silence_stop");
        }
        if self.preroll >= self.max_file {
            return err("preroll must be shorter than max_file");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn rejects_inverted_hysteresis_and_impossible_onset() {
        let c = Config { speech_exit: 0.6, ..Config::default() };
        assert!(c.validate().is_err());
        let c = Config { onset_min_speech: Duration::from_secs(3), ..Config::default() };
        assert!(c.validate().is_err());
        let c = Config { mic_silence_cap: Duration::from_secs(5), ..Config::default() };
        assert!(c.validate().is_err());
    }
}
