//! Scripted timelines for the recording controller.

use std::time::Duration;

use recorder_core::{Command, Config, Controller, Event, FrameObs, SessionId, State, StopReason, Stream};

const F: Duration = Duration::from_millis(32);
const SPEECH: (f32, f32) = (0.92, -22.0);
const QUIET: (f32, f32) = (0.03, -68.0);

fn secs(s: f64) -> Duration {
    Duration::from_secs_f64(s)
}

/// Drives a controller with 32 ms mic + loopback frames and records commands with timestamps.
struct Sim {
    c: Controller,
    now: Duration,
    log: Vec<(Duration, Command)>,
    buf: Vec<Command>,
}

impl Sim {
    fn new() -> Self {
        Self::with(Config::default())
    }

    fn with(cfg: Config) -> Self {
        let mut s =
            Self { c: Controller::new(cfg, F, F).unwrap(), now: Duration::ZERO, log: vec![], buf: vec![] };
        s.event(Event::SignedIn { at: Duration::ZERO });
        s
    }

    fn event(&mut self, ev: Event) {
        self.c.step(ev, &mut self.buf);
        let at = ev.at();
        self.log.extend(self.buf.drain(..).map(|c| (at, c)));
    }

    fn frames(&mut self, dur: Duration, mic: (f32, f32), loopback: (f32, f32)) {
        let end = self.now + dur;
        while self.now < end {
            let start = self.now;
            self.event(Event::Frame {
                stream: Stream::Mic,
                obs: FrameObs { start, len: F, prob: mic.0, rms_db: mic.1 },
            });
            self.event(Event::Frame {
                stream: Stream::Loopback,
                obs: FrameObs { start, len: F, prob: loopback.0, rms_db: loopback.1 },
            });
            self.now += F;
        }
    }

    fn speak(&mut self, s: f64) {
        self.frames(secs(s), SPEECH, QUIET);
    }
    fn quiet(&mut self, s: f64) {
        self.frames(secs(s), QUIET, QUIET);
    }
    fn remote_speaks(&mut self, s: f64) {
        self.frames(secs(s), QUIET, SPEECH);
    }

    fn starts(&self) -> Vec<(Duration, Duration, Duration, Option<SessionId>)> {
        self.log
            .iter()
            .filter_map(|(at, c)| match c {
                Command::StartFile { media_start, trigger, rotated_from, .. } => {
                    Some((*at, *media_start, *trigger, *rotated_from))
                }
                _ => None,
            })
            .collect()
    }

    fn stops(&self) -> Vec<(Duration, StopReason, bool)> {
        self.log
            .iter()
            .filter_map(|(_, c)| match c {
                Command::StopFile { end, reason, likely_false_trigger, .. } => {
                    Some((*end, *reason, *likely_false_trigger))
                }
                _ => None,
            })
            .collect()
    }
}

fn close(a: Duration, b: Duration, tol: Duration) -> bool {
    a.abs_diff(b) <= tol
}

#[test]
fn coughs_and_short_words_never_start_a_recording() {
    let mut s = Sim::new();
    s.quiet(5.0);
    for _ in 0..30 {
        s.speak(0.2);
        s.quiet(1.0);
    }
    assert!(s.starts().is_empty(), "{:?}", s.log);
}

#[test]
fn starts_one_second_into_speech_with_full_preroll() {
    let mut s = Sim::new();
    s.quiet(20.0);
    let speech_start = s.now;
    s.speak(3.0);
    let starts = s.starts();
    assert_eq!(starts.len(), 1);
    let (_, media_start, trigger, rotated) = starts[0];
    assert!(close(trigger, speech_start + secs(1.0), F), "trigger {trigger:?}");
    assert_eq!(media_start, trigger - Config::default().preroll);
    assert_eq!(rotated, None);
    assert_eq!(s.log[0].1, Command::OpenCamera);
}

#[test]
fn preroll_never_reaches_before_sign_in() {
    let mut s = Sim::new();
    s.quiet(1.0);
    s.speak(2.0);
    assert_eq!(s.starts()[0].1, Duration::ZERO);
}

#[test]
fn stops_thirty_seconds_after_the_last_speech() {
    let mut s = Sim::new();
    s.quiet(10.0);
    s.speak(5.0);
    let last_speech = s.now;
    s.quiet(40.0);
    let stops = s.stops();
    assert_eq!(stops.len(), 1);
    assert_eq!(stops[0].1, StopReason::Silence);
    assert!(close(stops[0].0, last_speech + secs(30.0), F * 2), "{:?}", stops[0]);
    assert_eq!(s.log.last().unwrap().1, Command::CloseCamera);
    assert!(matches!(s.c.state(), State::Listening(_)));
}

#[test]
fn pauses_shorter_than_thirty_seconds_stay_in_one_file() {
    let mut s = Sim::new();
    s.quiet(10.0);
    s.speak(5.0);
    s.quiet(25.0);
    s.speak(5.0);
    s.quiet(20.0);
    s.speak(2.0);
    s.quiet(35.0);
    assert_eq!(s.starts().len(), 1);
    assert_eq!(s.stops().len(), 1);
}

#[test]
fn each_conversation_gets_its_own_file_without_overlap() {
    let mut s = Sim::new();
    s.quiet(10.0);
    s.speak(6.0);
    s.quiet(32.0);
    s.quiet(3.0);
    s.speak(6.0);
    s.quiet(35.0);
    let starts = s.starts();
    let stops = s.stops();
    assert_eq!((starts.len(), stops.len()), (2, 2));
    // Second file's pre-roll can't reach back into the first file.
    assert!(starts[1].1 >= stops[0].0, "{starts:?} {stops:?}");
}

#[test]
fn remote_speech_alone_never_starts_but_keeps_a_recording_alive() {
    let mut s = Sim::new();
    s.quiet(5.0);
    s.remote_speaks(20.0);
    assert!(s.starts().is_empty());

    s.speak(3.0);
    assert_eq!(s.starts().len(), 1);
    // The patient on the call talks for 2 minutes while the doctor listens.
    s.remote_speaks(120.0);
    assert!(s.stops().is_empty(), "remote speech must keep the recording alive");
}

#[test]
fn mic_silence_cap_stops_music_or_video_holding_a_recording_open() {
    let mut s = Sim::new();
    s.quiet(5.0);
    s.speak(3.0);
    let last_mic = s.now;
    s.remote_speaks(6.0 * 60.0);
    let stops = s.stops();
    assert_eq!(stops.len(), 1);
    assert_eq!(stops[0].1, StopReason::MicSilenceCap);
    assert!(close(stops[0].0, last_mic + secs(300.0), F * 2));
}

#[test]
fn rotates_after_sixty_minutes_without_closing_the_camera() {
    let mut s = Sim::new();
    s.speak(61.0 * 60.0);
    let starts = s.starts();
    let stops = s.stops();
    assert_eq!(starts.len(), 2);
    assert_eq!(stops.len(), 1);
    assert_eq!(stops[0].1, StopReason::Rotation);
    let (_, second_media_start, _, rotated_from) = starts[1];
    assert_eq!(rotated_from, Some(SessionId(1)));
    assert_eq!(second_media_start, stops[0].0);
    assert!(close(stops[0].0 - starts[0].1, secs(3600.0), F));
    let cams: Vec<_> =
        s.log.iter().filter(|(_, c)| matches!(c, Command::OpenCamera | Command::CloseCamera)).collect();
    assert_eq!(cams.len(), 1, "rotation must not cycle the camera: {cams:?}");
}

#[test]
fn manual_stop_waits_for_quiet_before_re_arming() {
    let mut s = Sim::new();
    s.quiet(5.0);
    s.speak(5.0);
    s.event(Event::StopRequested { at: s.now });
    assert_eq!(s.stops()[0].1, StopReason::UserStop);
    s.speak(20.0); // chatter continues after the user stopped the recording
    assert_eq!(s.starts().len(), 1);
    s.quiet(10.5);
    s.speak(2.0);
    assert_eq!(s.starts().len(), 2);
}

#[test]
fn pause_stops_and_suppresses_until_it_expires() {
    let mut s = Sim::new();
    s.quiet(5.0);
    s.speak(3.0);
    let until = s.now + secs(60.0);
    s.event(Event::Pause { at: s.now, until: Some(until) });
    assert_eq!(s.stops()[0].1, StopReason::Paused);
    s.speak(50.0);
    assert_eq!(s.starts().len(), 1);
    s.quiet(11.0); // passes `until`
    assert!(matches!(s.c.state(), State::Listening(_)));
    s.speak(2.0);
    assert_eq!(s.starts().len(), 2);
}

#[test]
fn sign_out_finalizes_and_ignores_audio_until_sign_in() {
    let mut s = Sim::new();
    s.quiet(5.0);
    s.speak(3.0);
    s.event(Event::SignedOut { at: s.now });
    assert_eq!(s.stops()[0].1, StopReason::SignedOut);
    let n = s.log.len();
    s.speak(10.0);
    assert_eq!(s.log.len(), n, "no commands while signed out");
    s.event(Event::SignedIn { at: s.now });
    s.speak(2.0);
    assert_eq!(s.starts().len(), 2);
}

#[test]
fn sleep_finalizes_and_wake_starts_fresh() {
    let mut s = Sim::new();
    s.quiet(5.0);
    s.speak(3.0);
    s.event(Event::Sleep { at: s.now });
    assert_eq!(s.stops()[0].1, StopReason::Sleep);
    assert_eq!(*s.c.state(), State::Suspended);
    s.now += secs(3600.0);
    let woke = s.now;
    s.event(Event::Wake { at: woke });
    s.speak(2.0);
    let starts = s.starts();
    assert_eq!(starts.len(), 2);
    assert_eq!(starts[1].1, woke, "pre-roll must not span the sleep");
}

#[test]
fn ticks_enforce_the_timeout_when_audio_stops_arriving() {
    let mut s = Sim::new();
    s.quiet(5.0);
    s.speak(4.0);
    let last = s.now;
    for i in 1..=40 {
        s.event(Event::Tick { at: last + secs(f64::from(i)) });
    }
    let stops = s.stops();
    assert_eq!(stops.len(), 1);
    assert_eq!(stops[0].0, last + secs(30.0));
}

#[test]
fn short_recordings_are_flagged_long_ones_are_not() {
    let mut s = Sim::new();
    s.quiet(5.0);
    s.speak(1.3);
    s.quiet(35.0);
    s.speak(8.0);
    s.quiet(35.0);
    let flags: Vec<bool> = s.stops().iter().map(|s| s.2).collect();
    assert_eq!(flags, vec![true, false]);
}
