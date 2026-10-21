//! Property tests: invariants of the recording controller over random timelines.
//!
//! Inputs are generated so that "speech" frames are unambiguous (high VAD
//! probability, well above background) and everything else is clearly quiet.
//! The checker then verifies safety and liveness rules independently of the
//! controller's internals.

use std::collections::VecDeque;
use std::time::Duration;

use proptest::prelude::*;
use recorder_core::{Command, Config, Controller, Event, FrameObs, SessionId, State, StopReason, Stream};

const F: Duration = Duration::from_millis(32);
const HIGH: (f32, f32) = (0.92, -22.0);
const LOW: (f32, f32) = (0.03, -68.0);

/// Compressed timings so caps, rotations and re-arming happen within short runs.
fn cfg() -> Config {
    Config {
        silence_stop: Duration::from_secs(10),
        mic_silence_cap: Duration::from_secs(40),
        max_file: Duration::from_secs(90),
        preroll: Duration::from_secs(4),
        rearm_silence: Duration::from_secs(3),
        ..Config::default()
    }
}

#[derive(Debug, Clone)]
enum Step {
    Mic(u32),
    Quiet(u32),
    Remote(u32),
    Both(u32),
    Blip(u32),
    StopRequested,
    Pause(Option<u32>),
    Resume,
    SignOut,
    SignIn,
    Sleep,
    Wake(u32),
    /// Audio stops arriving; only clock ticks (device unplugged).
    TicksOnly(u32),
}

fn step() -> impl Strategy<Value = Step> {
    prop_oneof![
        6 => (10u32..600).prop_map(Step::Mic),
        6 => (1u32..500).prop_map(Step::Quiet),
        3 => (10u32..1500).prop_map(Step::Remote),
        1 => (10u32..300).prop_map(Step::Both),
        3 => (1u32..9).prop_map(Step::Blip),
        1 => Just(Step::StopRequested),
        1 => proptest::option::of(1u32..30).prop_map(Step::Pause),
        1 => Just(Step::Resume),
        1 => Just(Step::SignOut),
        2 => Just(Step::SignIn),
        1 => Just(Step::Sleep),
        1 => (1u32..120).prop_map(Step::Wake),
        1 => (1u32..60).prop_map(Step::TicksOnly),
    ]
}

#[derive(Default)]
struct Checker {
    open: Option<(SessionId, Duration, Duration)>, // id, media_start, trigger
    camera: bool,
    last_end: Option<Duration>,
    mic_high_ends: VecDeque<Duration>,
    last_high: Option<Duration>,
    mic_run: u32,
    loop_run: u32,
    last_long_mic_run: Option<Duration>,
    last_long_any_run: Option<Duration>,
    prev_frame_end: Option<Duration>,
    signed_in: bool,
}

struct Run {
    c: Controller,
    cfg: Config,
    now: Duration,
    chk: Checker,
    buf: Vec<Command>,
}

impl Run {
    fn new() -> Self {
        let cfg = cfg();
        Self {
            c: Controller::new(cfg.clone(), F, F).unwrap(),
            cfg,
            now: Duration::ZERO,
            chk: Checker::default(),
            buf: vec![],
        }
    }

    fn event(&mut self, ev: Event) -> Result<(), TestCaseError> {
        let was_signed_in = self.chk.signed_in;
        self.c.step(ev, &mut self.buf);
        let cmds = std::mem::take(&mut self.buf);
        if !was_signed_in && !matches!(ev, Event::SignedIn { .. }) {
            prop_assert!(cmds.is_empty(), "commands while signed out: {cmds:?}");
        }
        match ev {
            Event::SignedIn { .. } => self.chk.signed_in = true,
            Event::SignedOut { .. } => self.chk.signed_in = false,
            _ => {}
        }
        let mut last_was_rotation = false;
        for cmd in cmds {
            match cmd {
                Command::OpenCamera => {
                    prop_assert!(!self.chk.camera, "camera opened twice");
                    self.chk.camera = true;
                }
                Command::CloseCamera => {
                    prop_assert!(self.chk.camera, "camera closed while closed");
                    prop_assert!(self.chk.open.is_none(), "camera closed with a file open");
                    self.chk.camera = false;
                }
                Command::StartFile { session, media_start, trigger, rotated_from } => {
                    prop_assert!(self.chk.open.is_none(), "file started while another is open");
                    prop_assert!(self.chk.camera, "file started without camera");
                    prop_assert!(media_start <= trigger);
                    match rotated_from {
                        Some(_) => {
                            prop_assert!(last_was_rotation);
                            prop_assert_eq!(
                                Some(media_start),
                                self.chk.last_end,
                                "rotation must be seamless"
                            );
                        }
                        None => {
                            if let Some(end) = self.chk.last_end {
                                prop_assert!(media_start >= end, "pre-roll overlaps previous file");
                            }
                            prop_assert!(trigger - media_start <= self.cfg.preroll);
                            self.check_onset(trigger)?;
                        }
                    }
                    self.chk.open = Some((session, media_start, trigger));
                }
                Command::StopFile { session, end, reason, .. } => {
                    let Some((open_id, media_start, trigger)) = self.chk.open.take() else {
                        return Err(TestCaseError::fail("stop without open file"));
                    };
                    prop_assert_eq!(session, open_id);
                    prop_assert!(end >= media_start);
                    prop_assert!(end - media_start <= self.cfg.max_file + F, "file longer than max_file");
                    match reason {
                        StopReason::Silence => self.check_no_recent_long_run(
                            end,
                            trigger,
                            self.chk.last_long_any_run,
                            self.cfg.silence_stop,
                        )?,
                        StopReason::MicSilenceCap => self.check_no_recent_long_run(
                            end,
                            trigger,
                            self.chk.last_long_mic_run,
                            self.cfg.mic_silence_cap,
                        )?,
                        _ => {}
                    }
                    last_was_rotation = reason == StopReason::Rotation;
                    self.chk.last_end = Some(end);
                }
            }
        }
        prop_assert_eq!(self.chk.camera, self.chk.open.is_some(), "camera state must follow file state");
        Ok(())
    }

    /// START needs >= onset_min of speech frames inside the onset window ending at the trigger.
    fn check_onset(&self, trigger: Duration) -> Result<(), TestCaseError> {
        let from = trigger.saturating_sub(self.cfg.onset_window);
        let n = self.chk.mic_high_ends.iter().filter(|&&t| t > from && t <= trigger).count() as u32;
        prop_assert!(F * n >= self.cfg.onset_min_speech, "started with only {n} speech frames in window");
        Ok(())
    }

    /// A silence-type stop must not happen while a clearly keep-alive-worthy run is recent.
    fn check_no_recent_long_run(
        &self,
        end: Duration,
        trigger: Duration,
        last_run: Option<Duration>,
        timeout: Duration,
    ) -> Result<(), TestCaseError> {
        if let Some(run_end) = last_run.filter(|&t| t > trigger) {
            prop_assert!(
                end >= run_end + timeout - F,
                "stopped {:?} after a long speech run (timeout {timeout:?})",
                end - run_end
            );
        }
        Ok(())
    }

    /// Liveness: once nothing high has been heard for the timeout, no file may stay open.
    fn check_liveness(&self) -> Result<(), TestCaseError> {
        if let Some((_, _, trigger)) = self.chk.open {
            let reference = self.chk.last_high.map_or(trigger, |h| h.max(trigger));
            prop_assert!(
                self.now.saturating_sub(reference) <= self.cfg.silence_stop + F * 2,
                "still recording {:?} after the last speech",
                self.now - reference
            );
        }
        Ok(())
    }

    fn frames(&mut self, n: u32, mic: (f32, f32), lb: (f32, f32)) -> Result<(), TestCaseError> {
        for _ in 0..n {
            let start = self.now;
            let end = start + F;
            // Record what the input contained *before* the controller reacts to it.
            let (mic_hi, lb_hi) = (mic == HIGH, lb == HIGH);
            // A gap in the frame stream breaks any "consecutive" run.
            if self.chk.prev_frame_end != Some(start) {
                self.chk.mic_run = 0;
                self.chk.loop_run = 0;
            }
            self.chk.prev_frame_end = Some(end);
            if self.chk.signed_in {
                if mic_hi {
                    self.chk.mic_high_ends.push_back(end);
                    if self.chk.mic_high_ends.len() > 200 {
                        self.chk.mic_high_ends.pop_front();
                    }
                }
                if mic_hi || lb_hi {
                    self.chk.last_high = Some(end);
                }
                self.chk.mic_run = if mic_hi { self.chk.mic_run + 1 } else { 0 };
                self.chk.loop_run = if lb_hi { self.chk.loop_run + 1 } else { 0 };
                // Strictly consecutive runs comfortably above the keep-alive minimums.
                if self.chk.mic_run >= 11 {
                    self.chk.last_long_mic_run = Some(end);
                    self.chk.last_long_any_run = Some(end);
                }
                if self.chk.loop_run >= 17 {
                    self.chk.last_long_any_run = Some(end);
                }
            }
            self.event(Event::Frame {
                stream: Stream::Mic,
                obs: FrameObs { start, len: F, prob: mic.0, rms_db: mic.1 },
            })?;
            self.event(Event::Frame {
                stream: Stream::Loopback,
                obs: FrameObs { start, len: F, prob: lb.0, rms_db: lb.1 },
            })?;
            self.now = end;
            if self.chk.signed_in {
                self.check_liveness()?;
            }
        }
        Ok(())
    }

    /// Forget activity history across sign-out/sleep: detectors restart from scratch.
    fn reset_history(&mut self) {
        self.chk.mic_run = 0;
        self.chk.loop_run = 0;
        self.chk.last_high = None;
        self.chk.mic_high_ends.clear();
    }

    fn apply(&mut self, s: &Step) -> Result<(), TestCaseError> {
        let at = self.now;
        match *s {
            Step::Mic(n) => self.frames(n, HIGH, LOW)?,
            Step::Quiet(n) => self.frames(n, LOW, LOW)?,
            Step::Remote(n) => self.frames(n, LOW, HIGH)?,
            Step::Both(n) => self.frames(n, HIGH, HIGH)?,
            Step::Blip(n) => self.frames(n, HIGH, LOW)?,
            Step::StopRequested => self.event(Event::StopRequested { at })?,
            Step::Pause(until) => {
                self.event(Event::Pause { at, until: until.map(|s| at + Duration::from_secs(s.into())) })?
            }
            Step::Resume => self.event(Event::Resume { at })?,
            Step::SignOut => {
                self.event(Event::SignedOut { at })?;
                self.reset_history();
            }
            Step::SignIn => self.event(Event::SignedIn { at })?,
            Step::Sleep => self.event(Event::Sleep { at })?,
            Step::Wake(gap) => {
                let was_suspended = matches!(self.c.state(), State::Suspended);
                self.now += Duration::from_secs(gap.into());
                self.event(Event::Wake { at: self.now })?;
                if was_suspended {
                    self.reset_history();
                }
            }
            Step::TicksOnly(n) => {
                for _ in 0..n {
                    self.now += Duration::from_secs(1);
                    self.event(Event::Tick { at: self.now })?;
                    self.check_liveness()?;
                }
            }
        }
        Ok(())
    }
}

proptest! {
    // 256 cases by default; set PROPTEST_CASES for longer sweeps.
    #![proptest_config(ProptestConfig::default())]

    #[test]
    fn controller_invariants(steps in proptest::collection::vec(step(), 1..40)) {
        let mut run = Run::new();
        run.event(Event::SignedIn { at: Duration::ZERO })?;
        for s in &steps {
            run.apply(s)?;
        }
        // Signing out always leaves no file and no camera open.
        let at = run.now;
        run.event(Event::SignedOut { at })?;
        prop_assert!(run.chk.open.is_none() && !run.chk.camera);
    }
}
