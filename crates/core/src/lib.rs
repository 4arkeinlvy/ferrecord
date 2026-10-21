//! Pure decision logic for the recorder: no I/O, no threads, no wall clock.
//!
//! The shell feeds [`controller::Event`]s (VAD frames, ticks, user and OS
//! actions) with explicit timestamps and executes the returned
//! [`controller::Command`]s. Everything here is deterministic, so it is tested
//! with scripted timelines and property tests instead of real devices.
#![forbid(unsafe_code)]

pub mod config;
pub mod controller;
pub mod detector;
pub mod noise_floor;

pub use config::{Config, ConfigError};
pub use controller::{Command, Controller, Event, SessionId, State, StopReason, Stream};
pub use detector::{FrameObs, FrameVerdict};
