//! Offline evaluation of conversation detection.
//!
//! ```text
//! vad-eval run   [--engine silero|earshot|both] <file.wav>...
//! vad-eval synth --speech <speech.wav> --out <out.wav> [--minutes 30] [--conversations 8]
//!                [--noise-dbfs -55] [--speech-dbfs -26] [--distractors 40] [--seed 1]
//! vad-eval trace [--engine silero|earshot] <file.wav>        # per-frame CSV
//! vad-eval mix   [--raw testdata/corpus/raw] [--out testdata/corpus/real] [--streams 8]
//!                [--minutes 30] [--seed 7]                    # real-corpus streams (see mix.rs)
//! ```
//!
//! `run --frontend raw|hp|hp+ns` picks the audio front end before the VAD:
//! `hp+ns` (default, as the app does) is a 90 Hz high-pass plus RNNoise noise
//! suppression at 48 kHz; `hp` is the high-pass only; `raw` feeds samples unchanged.
//!
//! `run` accepts `--set key=value` (repeatable) to override detector settings:
//! `speech_enter`, `speech_exit`, `energy_margin_db`, `onset_min_speech_ms`,
//! `onset_window_ms`, `silence_stop_ms`, `preroll_ms`.
//!
//! Input is 16 kHz mono WAV (convert with `ffmpeg -i in -ac 1 -ar 16000 out.wav`).
//! Ground truth, if present, lives next to the audio as `<file>.labels.csv` with
//! one conversation per line: `start_sec,end_sec`.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use recorder_core::{Command, Config, Controller, Event, FrameObs, StopReason, Stream};
use recorder_vad::{EngineKind, SAMPLE_RATE, new_engine, rms_dbfs};

mod mix;

type Res<T> = Result<T, Box<dyn Error>>;

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Res<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: vad-eval <run|synth|trace> ... (see source header)".into());
    }
    let cmd = args.remove(0);
    match cmd.as_str() {
        "run" => cmd_run(args),
        "synth" => cmd_synth(args),
        "trace" => cmd_trace(args),
        "mix" => mix::cmd_mix(args),
        "scan" => cmd_scan(args),
        other => Err(format!("unknown command `{other}`").into()),
    }
}

/// Removes `--name value` from `args` and returns the value.
fn take_opt(args: &mut Vec<String>, name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    if i + 1 >= args.len() {
        return None;
    }
    let v = args.remove(i + 1);
    args.remove(i);
    Some(v)
}

fn engines_from(arg: Option<String>) -> Res<Vec<EngineKind>> {
    match arg.as_deref() {
        None | Some("silero") => Ok(vec![EngineKind::Silero]),
        Some("both") => Ok(EngineKind::ALL.to_vec()),
        Some(other) => Ok(vec![other.parse::<EngineKind>()?]),
    }
}

// ---------------------------------------------------------------------------
// Audio I/O
// ---------------------------------------------------------------------------

fn read_wav_16k(path: &Path) -> Res<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE || spec.channels != 1 {
        return Err(format!(
            "{}: need 16 kHz mono, got {} Hz x{} (ffmpeg -i in -ac 1 -ar 16000 out.wav)",
            path.display(),
            spec.sample_rate,
            spec.channels
        )
        .into());
    }
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<Vec<_>, _>>()?
        }
    };
    Ok(samples)
}

fn labels_path(wav: &Path) -> PathBuf {
    let mut p = wav.as_os_str().to_owned();
    p.push(".labels.csv");
    PathBuf::from(p)
}

#[derive(Debug, Clone, Copy)]
struct Span {
    start: Duration,
    end: Duration,
}

fn read_labels(wav: &Path) -> Res<Option<Vec<Span>>> {
    read_spans(&labels_path(wav))
}

fn distractors_path(wav: &Path) -> PathBuf {
    let mut p = wav.as_os_str().to_owned();
    p.push(".distractors.csv");
    PathBuf::from(p)
}

/// `start_sec,end_sec[,anything]` lines; `#` comments.
fn read_spans(path: &Path) -> Res<Option<Vec<Span>>> {
    if !path.exists() {
        return Ok(None);
    }
    let mut spans = Vec::new();
    for (n, line) in std::fs::read_to_string(path)?.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split(',').map(str::trim);
        let parse = |v: Option<&str>| -> Res<Duration> {
            Ok(Duration::from_secs_f64(v.ok_or("missing field")?.parse::<f64>()?))
        };
        let (start, end) = (parse(it.next())?, parse(it.next())?);
        if end <= start {
            return Err(format!("{}:{}: end must be after start", path.display(), n + 1).into());
        }
        spans.push(Span { start, end });
    }
    spans.sort_by_key(|s| s.start);
    Ok(Some(spans))
}

// ---------------------------------------------------------------------------
// run: engine + controller over a file
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Recording {
    media_start: Duration,
    trigger: Duration,
    end: Duration,
    reason: StopReason,
    flagged: bool,
}

struct FileResult {
    audio: Duration,
    engine_time: Duration,
    recordings: Vec<Recording>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Frontend {
    Raw,
    HighPass,
    HighPassNoiseSuppression,
}

impl std::str::FromStr for Frontend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "raw" => Ok(Frontend::Raw),
            "hp" => Ok(Frontend::HighPass),
            "hp+ns" => Ok(Frontend::HighPassNoiseSuppression),
            other => Err(format!("unknown frontend `{other}` (raw, hp, hp+ns)")),
        }
    }
}

/// The same processing the app applies before the VAD (offline, whole file).
fn apply_frontend(samples: &[f32], frontend: Frontend) -> Res<Vec<f32>> {
    let mut x = samples.to_vec();
    if frontend == Frontend::Raw {
        return Ok(x);
    }
    recorder_audio::HighPass::new(SAMPLE_RATE as f32, 90.0).process(&mut x);
    if frontend == Frontend::HighPass {
        return Ok(x);
    }
    // Same chain as the app: noise suppression at 48 kHz, then down to 16 kHz.
    let mut up = recorder_audio::StreamResampler::new(SAMPLE_RATE, recorder_audio::NS_RATE)?;
    let mut down = recorder_audio::StreamResampler::new(recorder_audio::NS_RATE, SAMPLE_RATE)?;
    let mut ns = recorder_audio::NoiseSuppressor::new();
    let mut out = Vec::with_capacity(x.len());
    let mut err: Option<String> = None;
    for block in x.chunks(4800) {
        up.push(block, |b48| {
            ns.push(b48, |f| {
                if let Err(e) = down.push(f, |b16| out.extend_from_slice(b16)) {
                    err.get_or_insert(e.to_string());
                }
            });
        })?;
    }
    if let Some(e) = err {
        return Err(e.into());
    }
    Ok(out)
}

fn analyse(samples: &[f32], kind: EngineKind, cfg: &Config) -> Res<FileResult> {
    let mut vad = new_engine(kind)?;
    let frame = vad.frame_duration();
    let mut ctl = Controller::new(cfg.clone(), frame, frame)?;
    let mut out = Vec::new();
    let mut open: Option<(Duration, Duration)> = None;
    let mut recordings = Vec::new();
    let mut engine_time = Duration::ZERO;

    let mut handle = |out: &mut Vec<Command>, open: &mut Option<(Duration, Duration)>| {
        for c in out.drain(..) {
            match c {
                Command::StartFile { media_start, trigger, .. } => *open = Some((media_start, trigger)),
                Command::StopFile { end, reason, likely_false_trigger, .. } => {
                    if let Some((media_start, trigger)) = open.take() {
                        recordings.push(Recording {
                            media_start,
                            trigger,
                            end,
                            reason,
                            flagged: likely_false_trigger,
                        });
                    }
                }
                Command::OpenCamera | Command::CloseCamera => {}
            }
        }
    };

    ctl.step(Event::SignedIn { at: Duration::ZERO }, &mut out);
    let n = vad.frame_len();
    for (i, chunk) in samples.chunks_exact(n).enumerate() {
        let t0 = Instant::now();
        let prob = vad.process(chunk)?;
        engine_time += t0.elapsed();
        let obs = FrameObs { start: frame * i as u32, len: frame, prob, rms_db: rms_dbfs(chunk) };
        ctl.step(Event::Frame { stream: Stream::Mic, obs }, &mut out);
        handle(&mut out, &mut open);
    }
    let audio = Duration::from_secs_f64(samples.len() as f64 / f64::from(SAMPLE_RATE));
    // Let pending silence timeouts fire as if audio continued quietly, then sign out.
    ctl.step(Event::Tick { at: audio + cfg.silence_stop }, &mut out);
    ctl.step(Event::SignedOut { at: audio + cfg.silence_stop }, &mut out);
    handle(&mut out, &mut open);
    Ok(FileResult { audio, engine_time, recordings })
}

#[derive(Default)]
struct Metrics {
    files: usize,
    audio: Duration,
    engine_time: Duration,
    recordings: usize,
    flagged: usize,
    conversations: usize,
    missed: usize,
    late: usize,
    first_words_lost: usize,
    false_triggers: usize,
    false_in_music: usize,
    non_conversation_audio: Duration,
    onset_latencies: Vec<Duration>,
    covered: Duration,
    labelled: Duration,
}

const MISS_WINDOW: Duration = Duration::from_secs(5);
const FALSE_TRIGGER_MARGIN: Duration = Duration::from_secs(2);

fn score(m: &mut Metrics, r: &FileResult, labels: Option<&[Span]>, distractors: &[Span]) {
    m.files += 1;
    m.audio += r.audio;
    m.engine_time += r.engine_time;
    m.recordings += r.recordings.len();
    m.flagged += r.recordings.iter().filter(|r| r.flagged).count();
    let Some(labels) = labels else { return };
    m.conversations += labels.len();
    let conv_total: Duration = labels.iter().map(|s| s.end - s.start).sum();
    m.labelled += conv_total;
    m.non_conversation_audio += r.audio.saturating_sub(conv_total);

    for conv in labels {
        // A recording whose media (pre-roll included) already covers the conversation start.
        let covering = r.recordings.iter().find(|rec| rec.media_start <= conv.start && rec.end > conv.start);
        // Otherwise the first recording that starts inside the conversation.
        let first_inside = r
            .recordings
            .iter()
            .filter(|rec| rec.trigger >= conv.start && rec.trigger < conv.end)
            .min_by_key(|rec| rec.trigger);
        match (covering, first_inside) {
            (Some(rec), _) => {
                if rec.trigger >= conv.start {
                    m.onset_latencies.push(rec.trigger - conv.start);
                }
            }
            (None, Some(rec)) => {
                let latency = rec.trigger - conv.start;
                m.onset_latencies.push(latency);
                m.first_words_lost += 1; // recorded, but the pre-roll didn't reach the first words
                if latency > MISS_WINDOW {
                    m.late += 1;
                }
            }
            (None, None) => m.missed += 1,
        }
        for rec in &r.recordings {
            let lo = rec.media_start.max(conv.start);
            let hi = rec.end.min(conv.end);
            if hi > lo {
                m.covered += hi - lo;
            }
        }
    }
    for rec in &r.recordings {
        let inside = labels.iter().any(|c| {
            rec.trigger + FALSE_TRIGGER_MARGIN >= c.start && rec.trigger <= c.end + FALSE_TRIGGER_MARGIN
        });
        if !inside {
            m.false_triggers += 1;
            let in_music = distractors.iter().any(|d| {
                rec.trigger + FALSE_TRIGGER_MARGIN >= d.start && rec.trigger <= d.end + FALSE_TRIGGER_MARGIN
            });
            if in_music {
                m.false_in_music += 1;
            }
        }
    }
}

fn percentile(v: &mut [Duration], p: f64) -> Option<Duration> {
    if v.is_empty() {
        return None;
    }
    v.sort();
    let idx = ((v.len() - 1) as f64 * p).round() as usize;
    Some(v[idx])
}

fn print_metrics(kind: EngineKind, m: &mut Metrics) {
    let hours = m.audio.as_secs_f64() / 3600.0;
    let rtf = m.engine_time.as_secs_f64() / m.audio.as_secs_f64().max(1e-9);
    println!("== {} ==", kind.name());
    println!("  audio            {:.1} min in {} file(s)", m.audio.as_secs_f64() / 60.0, m.files);
    println!(
        "  cpu              RTF {:.5}  ({:.1} s CPU per audio hour, one core)",
        rtf,
        m.engine_time.as_secs_f64() / hours.max(1e-9)
    );
    println!("  recordings       {} ({} flagged likely-false)", m.recordings, m.flagged);
    if m.conversations == 0 {
        println!("  (no labels: add <file>.labels.csv for accuracy metrics)");
        return;
    }
    let non_conv_h = m.non_conversation_audio.as_secs_f64() / 3600.0;
    let mean = if m.onset_latencies.is_empty() {
        None
    } else {
        Some(m.onset_latencies.iter().sum::<Duration>() / m.onset_latencies.len() as u32)
    };
    let p95 = percentile(&mut m.onset_latencies, 0.95);
    println!(
        "  conversations    {}  never recorded {}  late (>5 s, first words lost) {}  first-words-lost total {}",
        m.conversations, m.missed, m.late, m.first_words_lost
    );
    println!(
        "  onset latency    mean {}  p95 {}",
        mean.map_or("-".into(), |d| format!("{:.2}s", d.as_secs_f64())),
        p95.map_or("-".into(), |d| format!("{:.2}s", d.as_secs_f64()))
    );
    println!(
        "  false triggers   {}  ({:.2} per hour of non-conversation audio; {} inside music clips)",
        m.false_triggers,
        m.false_triggers as f64 / non_conv_h.max(1e-9),
        m.false_in_music
    );
    println!(
        "  coverage         {:.1}% of labelled conversation time recorded",
        100.0 * m.covered.as_secs_f64() / m.labelled.as_secs_f64().max(1e-9)
    );
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

fn cmd_run(mut args: Vec<String>) -> Res<()> {
    let engines = engines_from(take_opt(&mut args, "--engine"))?;
    let frontend: Frontend = take_opt(&mut args, "--frontend").as_deref().unwrap_or("hp+ns").parse()?;
    let mut cfg = Config::default();
    while let Some(kv) = take_opt(&mut args, "--set") {
        apply_override(&mut cfg, &kv)?;
    }
    cfg.validate()?;
    if args.is_empty() {
        return Err("run: give one or more 16 kHz mono WAV files".into());
    }
    type Loaded = (PathBuf, Vec<f32>, Option<Vec<Span>>, Vec<Span>);
    let files: Vec<Loaded> = args
        .iter()
        .map(|f| {
            let p = PathBuf::from(f);
            let samples = apply_frontend(&read_wav_16k(&p)?, frontend)?;
            let labels = read_labels(&p)?;
            let distractors = read_spans(&distractors_path(&p))?.unwrap_or_default();
            Ok((p, samples, labels, distractors))
        })
        .collect::<Res<_>>()?;
    for kind in engines {
        let mut m = Metrics::default();
        for (path, samples, labels, distractors) in &files {
            let r = analyse(samples, kind, &cfg)?;
            if std::env::var_os("VAD_EVAL_VERBOSE").is_some() {
                for rec in &r.recordings {
                    println!(
                        "  {}  rec {:>8.2}s..{:>8.2}s trigger {:>8.2}s {:?}{}",
                        path.display(),
                        rec.media_start.as_secs_f64(),
                        rec.end.as_secs_f64(),
                        rec.trigger.as_secs_f64(),
                        rec.reason,
                        if rec.flagged { " [flagged]" } else { "" }
                    );
                }
            }
            score(&mut m, &r, labels.as_deref(), distractors);
        }
        print_metrics(kind, &mut m);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// scan: how speech-like is each file (e.g. MUSAN noise beds)?
// ---------------------------------------------------------------------------

fn cmd_scan(mut args: Vec<String>) -> Res<()> {
    let engines = engines_from(take_opt(&mut args, "--engine"))?;
    let mut files: Vec<PathBuf> = Vec::new();
    for a in &args {
        let p = PathBuf::from(a);
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(&p)?
                .flatten()
                .map(|e| e.path())
                .filter(|q| q.extension().is_some_and(|x| x == "wav"))
                .collect();
            v.sort();
            files.extend(v);
        } else {
            files.push(p);
        }
    }
    let names: Vec<&str> = engines.iter().map(|k| k.name()).collect();
    println!(
        "file,seconds,{}",
        names.iter().map(|n| format!("{n}_speech_frac,{n}_longest_run_s")).collect::<Vec<_>>().join(",")
    );
    for f in files {
        let samples = read_wav_16k(&f)?;
        let mut cols = vec![
            f.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
            format!("{:.1}", samples.len() as f64 / f64::from(SAMPLE_RATE)),
        ];
        for &kind in &engines {
            let mut vad = new_engine(kind)?;
            let n = vad.frame_len();
            let (mut speech, mut total, mut run, mut longest) = (0usize, 0usize, 0usize, 0usize);
            for chunk in samples.chunks_exact(n) {
                total += 1;
                if vad.process(chunk)? >= 0.5 {
                    speech += 1;
                    run += 1;
                    longest = longest.max(run);
                } else {
                    run = 0;
                }
            }
            let frame_s = vad.frame_duration().as_secs_f64();
            cols.push(format!("{:.3}", speech as f64 / total.max(1) as f64));
            cols.push(format!("{:.1}", longest as f64 * frame_s));
        }
        println!("{}", cols.join(","));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// trace: per-frame CSV for debugging thresholds
// ---------------------------------------------------------------------------

fn cmd_trace(mut args: Vec<String>) -> Res<()> {
    let kind = engines_from(take_opt(&mut args, "--engine"))?[0];
    let path = PathBuf::from(args.first().ok_or("trace: give a WAV file")?);
    let samples = read_wav_16k(&path)?;
    let mut vad = new_engine(kind)?;
    let frame = vad.frame_duration();
    println!("t_sec,prob,rms_db");
    for (i, chunk) in samples.chunks_exact(vad.frame_len()).enumerate() {
        let p = vad.process(chunk)?;
        println!("{:.3},{:.4},{:.1}", (frame * i as u32).as_secs_f64(), p, rms_dbfs(chunk));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// synth: labelled test stream from a speech file + noise + distractors
// ---------------------------------------------------------------------------

fn db_to_amp(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// Pink-ish background noise (one-pole lowpassed white noise), unit RMS-ish.
struct Noise {
    rng: fastrand::Rng,
    state: f32,
}

impl Noise {
    fn sample(&mut self) -> f32 {
        let white = self.rng.f32() * 2.0 - 1.0;
        self.state = 0.97 * self.state + 0.03 * white;
        self.state * 9.0
    }
}

#[derive(Debug, Clone, Copy)]
enum Distractor {
    /// Cough / door: short broadband burst.
    Burst { start: usize, len: usize },
    /// Phone or notification beep: short tone.
    Beep { start: usize, len: usize, freq: f32 },
}

fn cmd_synth(mut args: Vec<String>) -> Res<()> {
    let speech_path = PathBuf::from(take_opt(&mut args, "--speech").ok_or("synth: --speech <wav> required")?);
    let out = PathBuf::from(take_opt(&mut args, "--out").ok_or("synth: --out <wav> required")?);
    let num = |args: &mut Vec<String>, n: &str, d: f64| -> Res<f64> {
        Ok(take_opt(args, n).map(|v| v.parse::<f64>()).transpose()?.unwrap_or(d))
    };
    let minutes = num(&mut args, "--minutes", 30.0)?;
    let conversations = num(&mut args, "--conversations", 8.0)? as usize;
    let noise_db = num(&mut args, "--noise-dbfs", -55.0)? as f32;
    let speech_db = num(&mut args, "--speech-dbfs", -26.0)? as f32;
    let distractors = num(&mut args, "--distractors", 40.0)? as usize;
    let seed = num(&mut args, "--seed", 1.0)? as u64;

    let speech = read_wav_16k(&speech_path)?;
    let speech_rms_db = rms_dbfs(&speech);
    let speech_gain = db_to_amp(speech_db - speech_rms_db);
    let sr = SAMPLE_RATE as usize;
    let total = (minutes * 60.0 * sr as f64) as usize;
    let mut rng = fastrand::Rng::with_seed(seed);

    // Conversations: 20-90 s slices of the speech file, at least 45 s apart so the
    // 30 s silence stop separates them.
    let min_gap = 45 * sr;
    let mut convs: Vec<(usize, usize, usize)> = Vec::new(); // start, len, offset in speech
    for _ in 0..conversations * 50 {
        if convs.len() == conversations {
            break;
        }
        let len = rng.usize(20 * sr..90 * sr).min(speech.len());
        if total <= len + 2 * min_gap {
            break;
        }
        let start = rng.usize(min_gap..total - len - min_gap);
        let clash = convs.iter().any(|&(s, l, _)| start < s + l + min_gap && s < start + len + min_gap);
        if !clash {
            convs.push((start, len, rng.usize(0..speech.len())));
        }
    }
    convs.sort();

    let mut extras = Vec::new();
    for _ in 0..distractors {
        let start = rng.usize(0..total.saturating_sub(sr).max(1));
        extras.push(if rng.bool() {
            Distractor::Burst { start, len: rng.usize(sr / 20..sr / 5) }
        } else {
            Distractor::Beep { start, len: rng.usize(sr / 10..sr * 3 / 5), freq: rng.f32() * 1500.0 + 600.0 }
        });
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&out, spec)?;
    let mut noise = Noise { rng: fastrand::Rng::with_seed(seed ^ 0x9e37), state: 0.0 };
    let noise_gain = db_to_amp(noise_db);
    let burst_gain = db_to_amp(-20.0);
    let beep_gain = db_to_amp(-24.0);
    for i in 0..total {
        let mut s = noise.sample() * noise_gain;
        if let Some(&(start, len, off)) = convs.iter().find(|&&(st, l, _)| i >= st && i < st + l) {
            s += speech[(off + i - start) % speech.len()] * speech_gain;
            let _ = len;
        }
        for d in &extras {
            match *d {
                Distractor::Burst { start, len } if i >= start && i < start + len => {
                    s += (noise.rng.f32() * 2.0 - 1.0) * burst_gain;
                }
                Distractor::Beep { start, len, freq } if i >= start && i < start + len => {
                    s += (2.0 * std::f32::consts::PI * freq * (i - start) as f32 / sr as f32).sin()
                        * beep_gain;
                }
                _ => {}
            }
        }
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;

    let mut labels = String::from("# start_sec,end_sec  (conversations that must be recorded)\n");
    for &(start, len, _) in &convs {
        labels.push_str(&format!(
            "{:.3},{:.3}\n",
            start as f64 / sr as f64,
            (start + len) as f64 / sr as f64
        ));
    }
    std::fs::write(labels_path(&out), labels)?;
    println!(
        "wrote {} ({:.1} min, {} conversations, {} distractors, noise {noise_db} dBFS, speech {speech_db} dBFS)",
        out.display(),
        minutes,
        convs.len(),
        extras.len()
    );
    Ok(())
}
