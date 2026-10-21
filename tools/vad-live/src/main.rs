//! Live microphone -> VAD -> recording controller, for tuning detection on a real mic.
//!
//! ```text
//! nix-shell --run "cargo run --release -p vad-live -- [--engine silero|earshot] [--device <name part>]
//!                  [--seconds N] [--set key=value]... | --list"
//! ```
//!
//! Runs the same chain the recorder will use (downmix -> 90 Hz high-pass ->
//! RNNoise at 48 kHz -> 16 kHz -> VAD -> controller) and prints START/STOP
//! decisions as they happen. `--no-ns` skips the noise suppression.
//! Nothing is recorded or written to disk.

use std::error::Error;
use std::io::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use recorder_audio::{HighPass, NS_RATE, NoiseSuppressor, StreamResampler, To16k, downmix};
use recorder_core::{Command, Config, Controller, Event, FrameObs, State, Stream};
use recorder_vad::{EngineKind, Framer, new_engine, rms_dbfs};

type Res<T> = Result<T, Box<dyn Error>>;

fn main() {
    if let Err(e) = run() {
        eprintln!("\nerror: {e}");
        std::process::exit(1);
    }
}

fn take_opt(args: &mut Vec<String>, name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    if i + 1 >= args.len() {
        return None;
    }
    let v = args.remove(i + 1);
    args.remove(i);
    Some(v)
}

fn apply_override(cfg: &mut Config, kv: &str) -> Res<()> {
    let (key, value) = kv.split_once('=').ok_or_else(|| format!("--set expects key=value, got `{kv}`"))?;
    let ms = |v: &str| -> Res<Duration> { Ok(Duration::from_millis(v.parse()?)) };
    match key {
        "speech_enter" => cfg.speech_enter = value.parse()?,
        "speech_exit" => cfg.speech_exit = value.parse()?,
        "energy_margin_db" => cfg.energy_margin_db = value.parse()?,
        "onset_min_speech_ms" => cfg.onset_min_speech = ms(value)?,
        "onset_window_ms" => cfg.onset_window = ms(value)?,
        "silence_stop_ms" => cfg.silence_stop = ms(value)?,
        "preroll_ms" => cfg.preroll = ms(value)?,
        other => return Err(format!("unknown setting `{other}`").into()),
    }
    Ok(())
}

fn device_name(d: &cpal::Device) -> String {
    d.description().map(|desc| desc.name().to_owned()).unwrap_or_else(|_| "<unnamed>".into())
}

fn run() -> Res<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let host = cpal::default_host();

    if args.iter().any(|a| a == "--list") {
        for d in host.input_devices()? {
            let cfg = d
                .default_input_config()
                .map(|c| format!("{} Hz x{} {:?}", c.sample_rate(), c.channels(), c.sample_format()));
            println!("{}  [{}]", device_name(&d), cfg.unwrap_or_else(|e| e.to_string()));
        }
        return Ok(());
    }

    let kind: EngineKind = take_opt(&mut args, "--engine").as_deref().unwrap_or("silero").parse()?;
    let seconds = take_opt(&mut args, "--seconds").map(|v| v.parse::<f64>()).transpose()?;
    let wanted = take_opt(&mut args, "--device");
    let use_ns = if let Some(i) = args.iter().position(|a| a == "--no-ns") {
        args.remove(i);
        false
    } else {
        true
    };
    let mut cfg = Config::default();
    while let Some(kv) = take_opt(&mut args, "--set") {
        apply_override(&mut cfg, &kv)?;
    }
    cfg.validate()?;
    if let Some(extra) = args.first() {
        return Err(format!("unexpected argument `{extra}`").into());
    }

    let device = match &wanted {
        Some(part) => host
            .input_devices()?
            .find(|d| device_name(d).to_lowercase().contains(&part.to_lowercase()))
            .ok_or_else(|| format!("no input device matching `{part}` (try --list)"))?,
        None => host.default_input_device().ok_or("no default input device")?,
    };
    let supported = device.default_input_config()?;
    let rate = supported.sample_rate();
    let channels = usize::from(supported.channels());
    let format = supported.sample_format();
    let stream_cfg = supported.config();

    // Real-time side: the callback only copies into a lock-free ring; never blocks.
    let (mut producer, mut consumer) = rtrb::RingBuffer::<f32>::new(rate as usize * channels * 2);
    let dropped = Arc::new(AtomicU64::new(0));
    let dropped_cb = Arc::clone(&dropped);
    let on_error = |e: cpal::Error| eprintln!("\naudio stream error: {e}");
    let stream = match format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            stream_cfg,
            move |data: &[f32], _: &cpal::InputCallbackInfo| match producer.write_chunk_uninit(data.len()) {
                Ok(chunk) => {
                    chunk.fill_from_iter(data.iter().copied());
                }
                Err(_) => {
                    dropped_cb.fetch_add(data.len() as u64, Ordering::Relaxed);
                }
            },
            on_error,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            stream_cfg,
            move |data: &[i16], _: &cpal::InputCallbackInfo| match producer.write_chunk_uninit(data.len()) {
                Ok(chunk) => {
                    chunk.fill_from_iter(data.iter().map(|&s| f32::from(s) / 32768.0));
                }
                Err(_) => {
                    dropped_cb.fetch_add(data.len() as u64, Ordering::Relaxed);
                }
            },
            on_error,
            None,
        )?,
        other => return Err(format!("unsupported input sample format {other:?}").into()),
    };
    stream.play()?;

    let mut engine = new_engine(kind)?;
    let frame = engine.frame_duration();
    let mut ctl = Controller::new(cfg.clone(), frame, frame)?;
    let mut out = Vec::new();
    ctl.step(Event::SignedIn { at: Duration::ZERO }, &mut out);

    println!(
        "listening on `{}` ({rate} Hz x{channels} {format:?}) with {}{} | start: >= {:.1}s speech in {:.1}s, stop: {:.0}s silence",
        device_name(&device),
        kind.name(),
        if use_ns { " + RNNoise" } else { "" },
        cfg.onset_min_speech.as_secs_f64(),
        cfg.onset_window.as_secs_f64(),
        cfg.silence_stop.as_secs_f64()
    );

    let mut highpass = HighPass::new(rate as f32, 90.0);
    let mut to48k = StreamResampler::new(rate, NS_RATE)?;
    let mut ns = NoiseSuppressor::new();
    let mut to16k = To16k::new(if use_ns { NS_RATE } else { rate })?;
    let mut denoised = Vec::with_capacity(NS_RATE as usize);
    let mut framer = Framer::new(engine.frame_len());
    let mut interleaved = Vec::with_capacity(rate as usize * channels);
    let mut mono = Vec::with_capacity(rate as usize);
    let mut frames: u32 = 0;
    let mut last = (0.0f32, -100.0f32);
    let started = Instant::now();
    let mut next_status = Instant::now();
    let mut failure: Option<String> = None;
    // Exact digital silence for 3 s almost always means a muted or disconnected mic.
    let mut silent_frames: u32 = 0;
    let silent_hint_after = (3_000_000 / frame.as_micros().max(1)) as u32;

    loop {
        if seconds.is_some_and(|s| started.elapsed().as_secs_f64() >= s) {
            break;
        }
        let available = consumer.slots();
        if available == 0 {
            std::thread::sleep(Duration::from_millis(10));
        } else {
            let chunk = consumer.read_chunk(available)?;
            let (a, b) = chunk.as_slices();
            interleaved.clear();
            interleaved.extend_from_slice(a);
            interleaved.extend_from_slice(b);
            chunk.commit_all();

            downmix(&interleaved, channels, &mut mono);
            highpass.process(&mut mono);
            let vad_input: &[f32] = if use_ns {
                denoised.clear();
                to48k.push(&mono, |b48| ns.push(b48, |f| denoised.extend_from_slice(f)))?;
                &denoised
            } else {
                &mono
            };
            to16k.push(vad_input, |block| {
                framer.push(block, |f| {
                    let prob = match engine.process(f) {
                        Ok(p) => p,
                        Err(e) => {
                            failure.get_or_insert(e.to_string());
                            return;
                        }
                    };
                    let rms_db = rms_dbfs(f);
                    last = (prob, rms_db);
                    if rms_db <= -100.0 {
                        silent_frames += 1;
                        if silent_frames == silent_hint_after {
                            println!(
                                "\n(input is pure digital silence: is the mic muted? \
                                 try `wpctl set-mute @DEFAULT_SOURCE@ 0` or --list / --device)"
                            );
                        }
                    } else {
                        silent_frames = 0;
                    }
                    let obs = FrameObs { start: frame * frames, len: frame, prob, rms_db };
                    frames += 1;
                    ctl.step(Event::Frame { stream: Stream::Mic, obs }, &mut out);
                    for cmd in out.drain(..) {
                        print_command(&cmd);
                    }
                });
            })?;
            if let Some(e) = failure.take() {
                return Err(e.into());
            }
        }
        if Instant::now() >= next_status {
            next_status += Duration::from_millis(200);
            print_status(frame * frames, last, ctl.state(), dropped.load(Ordering::Relaxed));
        }
    }
    let end = frame * frames;
    ctl.step(Event::SignedOut { at: end }, &mut out);
    println!();
    for cmd in out.drain(..) {
        print_command(&cmd);
    }
    Ok(())
}

fn print_status(t: Duration, (prob, rms_db): (f32, f32), state: &State, dropped: u64) {
    let bar: String = (0..20).map(|i| if (i as f32) < prob * 20.0 { '#' } else { '.' }).collect();
    let state = match state {
        State::Recording(s) => {
            format!("RECORDING #{} {:>5.1}s", s.id.0, (t.saturating_sub(s.media_start)).as_secs_f64())
        }
        State::Listening(l) if l.awaiting_quiet => "listening (waiting for quiet)".into(),
        State::Listening(_) => "listening".into(),
        State::Paused { .. } => "paused".into(),
        State::Suspended => "suspended".into(),
        State::SignedOut => "signed out".into(),
    };
    let drops = if dropped > 0 { format!("  dropped {dropped}") } else { String::new() };
    print!("\r{:>8.1}s  p={prob:.2} [{bar}] {rms_db:>6.1} dBFS  {state:<28}{drops}   ", t.as_secs_f64());
    let _ = std::io::stdout().flush();
}

fn print_command(cmd: &Command) {
    match cmd {
        Command::StartFile { session, media_start, trigger, rotated_from } => println!(
            "\n>>> START #{} at {:.2}s (file begins {:.2}s, pre-roll {:.1}s){}",
            session.0,
            trigger.as_secs_f64(),
            media_start.as_secs_f64(),
            (*trigger - *media_start).as_secs_f64(),
            rotated_from.map_or(String::new(), |r| format!(", continues #{}", r.0))
        ),
        Command::StopFile { session, end, reason, mic_speech, likely_false_trigger } => println!(
            "\n<<< STOP  #{} at {:.2}s ({reason:?}, {:.1}s of speech){}",
            session.0,
            end.as_secs_f64(),
            mic_speech.as_secs_f64(),
            if *likely_false_trigger { " [likely false trigger]" } else { "" }
        ),
        Command::OpenCamera | Command::CloseCamera => {}
    }
}
