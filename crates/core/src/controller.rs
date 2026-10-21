//! The recorder's brain: a pure state machine from detector output and user/OS
//! events to recording commands.
//!
//! Time comes only from event timestamps (a shared monotonic media timeline),
//! so every behaviour is reproducible in tests. The shell owns devices, files
//! and the network and simply executes [`Command`]s.

use std::time::Duration;

use crate::config::{Config, ConfigError};
use crate::detector::{FrameObs, StreamDetector, StreamParams, VoicedWindow};

/// Identifies one recording file. Monotonic within a controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    /// Room microphone (after echo cancellation). Starts and keeps recordings alive.
    Mic,
    /// System audio (remote side of a call). Only keeps an existing recording alive.
    Loopback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// No speech for `silence_stop`.
    Silence,
    /// Mic silent for `mic_silence_cap` even though system audio kept talking.
    MicSilenceCap,
    /// File reached `max_file`; a new file continues seamlessly.
    Rotation,
    /// "Stop current recording" from the tray.
    UserStop,
    Paused,
    SignedOut,
    Sleep,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    SignedIn {
        at: Duration,
    },
    SignedOut {
        at: Duration,
    },
    Frame {
        stream: Stream,
        obs: FrameObs,
    },
    /// Periodic clock tick so timeouts fire even if audio stops arriving.
    Tick {
        at: Duration,
    },
    /// Pause auto-recording until `until` (or until [`Event::Resume`]).
    Pause {
        at: Duration,
        until: Option<Duration>,
    },
    Resume {
        at: Duration,
    },
    StopRequested {
        at: Duration,
    },
    Sleep {
        at: Duration,
    },
    Wake {
        at: Duration,
    },
}

impl Event {
    /// When the event begins; deadlines up to this instant are applied first.
    pub fn start(&self) -> Duration {
        match *self {
            Event::Frame { obs, .. } => obs.start,
            _ => self.at(),
        }
    }

    /// When the event is complete (frame end for frames).
    pub fn at(&self) -> Duration {
        match *self {
            Event::Frame { obs, .. } => obs.end(),
            Event::SignedIn { at }
            | Event::SignedOut { at }
            | Event::Tick { at }
            | Event::Pause { at, .. }
            | Event::Resume { at }
            | Event::StopRequested { at }
            | Event::Sleep { at }
            | Event::Wake { at } => at,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    OpenCamera,
    CloseCamera,
    /// Open a new file whose media begins at `media_start` (pre-roll included).
    StartFile {
        session: SessionId,
        media_start: Duration,
        trigger: Duration,
        rotated_from: Option<SessionId>,
    },
    /// Finalize the file at `end`.
    StopFile {
        session: SessionId,
        end: Duration,
        reason: StopReason,
        mic_speech: Duration,
        likely_false_trigger: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Listening {
    /// Pre-roll never reaches back before this (previous file end, sign-in, wake).
    pub since: Duration,
    /// After a manual stop: suppress START until the mic has been quiet a while.
    pub awaiting_quiet: bool,
    pub last_mic_voiced: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Session {
    pub id: SessionId,
    pub media_start: Duration,
    pub trigger: Duration,
    pub last_keepalive: Duration,
    pub last_mic_keepalive: Duration,
    pub mic_speech: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub enum State {
    SignedOut,
    Listening(Listening),
    Recording(Session),
    Paused {
        until: Option<Duration>,
    },
    /// System sleep: no frames expected until [`Event::Wake`].
    Suspended,
}

pub struct Controller {
    cfg: Config,
    state: State,
    mic: StreamDetector,
    loopback: StreamDetector,
    window: VoicedWindow,
    next_id: u64,
}

impl Controller {
    pub fn new(cfg: Config, mic_frame: Duration, loopback_frame: Duration) -> Result<Self, ConfigError> {
        cfg.validate()?;
        if mic_frame.is_zero() || loopback_frame.is_zero() {
            return Err(ConfigError("frame durations must be positive".into()));
        }
        Ok(Self {
            mic: StreamDetector::new(StreamParams::mic(&cfg), mic_frame, &cfg),
            loopback: StreamDetector::new(StreamParams::loopback(&cfg), loopback_frame, &cfg),
            window: VoicedWindow::new(cfg.onset_window, mic_frame),
            state: State::SignedOut,
            next_id: 1,
            cfg,
        })
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn is_recording(&self) -> bool {
        matches!(self.state, State::Recording(_))
    }

    /// Advance the state machine. Commands are appended to `out` in execution order.
    pub fn step(&mut self, ev: Event, out: &mut Vec<Command>) {
        // Apply every deadline that passed before this event, at its exact time,
        // so gaps in the event stream can't stretch files or revive old sessions.
        self.advance_to(ev.start(), out);
        match ev {
            Event::SignedIn { at } => {
                if matches!(self.state, State::SignedOut) {
                    self.mic.reset();
                    self.loopback.reset();
                    self.enter_listening(at, false);
                }
            }
            Event::SignedOut { at } => {
                self.stop_recording(at, StopReason::SignedOut, out);
                self.state = State::SignedOut;
            }
            Event::Frame { stream, obs } => self.on_frame(stream, &obs, out),
            Event::Tick { .. } => {}
            Event::Pause { at, until } => {
                if !matches!(self.state, State::SignedOut | State::Suspended) {
                    self.stop_recording(at, StopReason::Paused, out);
                    self.state = State::Paused { until };
                }
            }
            Event::Resume { at } => {
                if matches!(self.state, State::Paused { .. }) {
                    self.enter_listening(at, false);
                }
            }
            Event::StopRequested { at } => {
                if self.stop_recording(at, StopReason::UserStop, out) {
                    self.enter_listening(at, true);
                }
            }
            Event::Sleep { at } => {
                if !matches!(self.state, State::SignedOut) {
                    self.stop_recording(at, StopReason::Sleep, out);
                    self.state = State::Suspended;
                }
            }
            Event::Wake { at } => {
                if matches!(self.state, State::Suspended) {
                    self.mic.reset();
                    self.loopback.reset();
                    self.enter_listening(at, false);
                }
            }
        }
    }

    fn on_frame(&mut self, stream: Stream, obs: &FrameObs, out: &mut Vec<Command>) {
        if matches!(self.state, State::SignedOut | State::Suspended) {
            return;
        }
        let verdict = match stream {
            Stream::Mic => self.mic.observe(obs),
            Stream::Loopback => self.loopback.observe(obs),
        };
        let now = obs.end();
        match &mut self.state {
            State::Listening(l) if stream == Stream::Mic => {
                if verdict.voiced {
                    l.last_mic_voiced = now;
                }
                if l.awaiting_quiet {
                    if now.saturating_sub(l.last_mic_voiced) >= self.cfg.rearm_silence {
                        l.awaiting_quiet = false;
                        self.window.clear();
                    }
                    return;
                }
                self.window.push(verdict.voiced, now);
                if self.window.voiced_duration() >= self.cfg.onset_min_speech {
                    self.start_recording(now, out);
                }
            }
            State::Recording(s) => {
                if stream == Stream::Mic && verdict.voiced {
                    s.mic_speech += obs.len;
                }
                if verdict.keepalive {
                    s.last_keepalive = s.last_keepalive.max(now);
                    if stream == Stream::Mic {
                        s.last_mic_keepalive = s.last_mic_keepalive.max(now);
                    }
                }
                self.advance_to(now, out);
            }
            State::Paused { .. } => self.advance_to(now, out),
            _ => {}
        }
    }

    /// Time-driven transitions up to `now`: silence stop, mic cap, rotation and
    /// pause expiry, each applied at its own deadline (several may be due).
    fn advance_to(&mut self, now: Duration, out: &mut Vec<Command>) {
        loop {
            match &self.state {
                State::Recording(s) => {
                    let silence = s.last_keepalive + self.cfg.silence_stop;
                    let cap = s.last_mic_keepalive + self.cfg.mic_silence_cap;
                    let rotation = s.media_start + self.cfg.max_file;
                    // Stops win ties with rotation: no point opening a file to close it.
                    let (deadline, reason) = [
                        (silence, StopReason::Silence),
                        (cap, StopReason::MicSilenceCap),
                        (rotation, StopReason::Rotation),
                    ]
                    .into_iter()
                    .min_by_key(|(t, _)| *t)
                    .expect("non-empty");
                    if deadline > now {
                        return;
                    }
                    if reason == StopReason::Rotation {
                        self.rotate(deadline, out);
                    } else {
                        self.stop_recording(deadline, reason, out);
                        self.enter_listening(deadline, false);
                    }
                }
                State::Paused { until: Some(until) } if *until <= now => {
                    let until = *until;
                    self.enter_listening(until, false);
                }
                _ => return,
            }
        }
    }

    fn next_session(&mut self) -> SessionId {
        let id = SessionId(self.next_id);
        self.next_id += 1;
        id
    }

    fn start_recording(&mut self, now: Duration, out: &mut Vec<Command>) {
        let since = match &self.state {
            State::Listening(l) => l.since,
            _ => now,
        };
        let id = self.next_session();
        let media_start = now.saturating_sub(self.cfg.preroll).max(since);
        out.push(Command::OpenCamera);
        out.push(Command::StartFile { session: id, media_start, trigger: now, rotated_from: None });
        self.state = State::Recording(Session {
            id,
            media_start,
            trigger: now,
            last_keepalive: now,
            last_mic_keepalive: now,
            mic_speech: self.window.voiced_duration(),
        });
        self.window.clear();
    }

    fn rotate(&mut self, now: Duration, out: &mut Vec<Command>) {
        let State::Recording(old) = &self.state else { return };
        let old = old.clone();
        out.push(self.stop_command(&old, now, StopReason::Rotation));
        let id = self.next_session();
        out.push(Command::StartFile {
            session: id,
            media_start: now,
            trigger: now,
            rotated_from: Some(old.id),
        });
        self.state = State::Recording(Session {
            id,
            media_start: now,
            trigger: now,
            last_keepalive: old.last_keepalive,
            last_mic_keepalive: old.last_mic_keepalive,
            mic_speech: Duration::ZERO,
        });
    }

    fn stop_command(&self, s: &Session, end: Duration, reason: StopReason) -> Command {
        Command::StopFile {
            session: s.id,
            end: end.max(s.media_start),
            reason,
            mic_speech: s.mic_speech,
            likely_false_trigger: s.mic_speech < self.cfg.likely_false_trigger_speech,
        }
    }

    /// Stops the current recording if any. Returns whether one was stopped.
    fn stop_recording(&mut self, now: Duration, reason: StopReason, out: &mut Vec<Command>) -> bool {
        let State::Recording(s) = &self.state else { return false };
        out.push(self.stop_command(s, now, reason));
        out.push(Command::CloseCamera);
        self.mic.reset_activity();
        self.loopback.reset_activity();
        true
    }

    fn enter_listening(&mut self, at: Duration, awaiting_quiet: bool) {
        self.window.clear();
        self.state = State::Listening(Listening { since: at, awaiting_quiet, last_mic_voiced: at });
    }
}
