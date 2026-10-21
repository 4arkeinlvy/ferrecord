//! `vad-eval mix`: labelled long streams from the open-license corpus.
//!
//! Layout expected under `--raw` (see `scripts/fetch-corpus.sh`):
//! - `ami/<meeting>.Array1-01.wav` + `ami/annotations/segments/<meeting>.<spk>.segments.xml`
//! - `fleurs/{dev,test}/*.wav` (Indonesian read speech)
//! - `musan/musan/noise/**.wav`, `musan/musan/music/**.wav`
//!
//! Each stream: a MUSAN noise bed, conversations at the stream's SNR (AMI
//! far-field excerpts or FLEURS-id "dialogues" with turn gaps and light
//! reverb), and unlabelled music clips as hard negatives. Writes
//! `<name>.wav`, `<name>.wav.labels.csv` and `<name>.wav.distractors.csv`.

use std::path::{Path, PathBuf};

use recorder_vad::SAMPLE_RATE;

use recorder_vad::rms_dbfs as rms_db_of;

use crate::{Res, labels_path, read_wav_16k};

const SR: usize = SAMPLE_RATE as usize;
/// Conversations are kept this far apart so the 30 s silence stop separates them.
const MIN_GAP_S: usize = 45;
const SNRS_DB: [f32; 4] = [20.0, 10.0, 5.0, 0.0];
const BED_DBFS: f32 = -45.0;

fn wavs_under(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "wav") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Speech segments (seconds) of one AMI meeting, all speakers merged.
fn ami_speech(segments_dir: &Path, meeting: &str) -> Vec<(f64, f64)> {
    let mut segs = Vec::new();
    for spk in ["A", "B", "C", "D", "E"] {
        let Ok(xml) = std::fs::read_to_string(segments_dir.join(format!("{meeting}.{spk}.segments.xml")))
        else {
            continue;
        };
        let attr = |tag: &str, s: &str| -> Option<f64> {
            let i = s.find(tag)? + tag.len();
            let rest = &s[i..];
            rest[..rest.find('"')?].parse().ok()
        };
        for line in xml.lines().filter(|l| l.contains("<segment ")) {
            if let (Some(a), Some(b)) = (attr("transcriber_start=\"", line), attr("transcriber_end=\"", line))
                && b > a
            {
                segs.push((a, b));
            }
        }
    }
    segs.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (a, b) in segs {
        match merged.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    merged
}

/// Which merged speech segments are clearly audible at the far-field mic:
/// median frame level at least 6 dB above the local room-noise floor (10th
/// percentile of frame levels within +-30 s). AMI annotations come from
/// close-talk headsets, so some annotated speech is near-inaudible at the array.
fn audible_segments(audio: &[f32], speech: &[(f64, f64)]) -> Vec<bool> {
    let hop = SR / 50; // 20 ms frames
    let levels: Vec<f32> = audio.chunks(hop).map(rms_db_of).collect();
    let frame = |t: f64| ((t * 50.0) as usize).min(levels.len().saturating_sub(1));
    speech
        .iter()
        .map(|&(a, b)| {
            let (fa, fb) = (frame(a), frame(b).max(frame(a) + 1));
            let mut seg: Vec<f32> = levels[fa..fb.min(levels.len())].to_vec();
            let mut around: Vec<f32> = levels[frame((a - 30.0).max(0.0))..frame(b + 30.0)].to_vec();
            if seg.is_empty() || around.is_empty() {
                return false;
            }
            seg.sort_by(f32::total_cmp);
            around.sort_by(f32::total_cmp);
            seg[seg.len() / 2] >= around[around.len() / 10] + 6.0
        })
        .collect()
}

/// An excerpt that starts at an *audible* speech onset after >= 3 s of quiet
/// and has no internal pause longer than 12 s. Returns (start_s, end_s).
fn ami_excerpt(
    speech: &[(f64, f64)],
    audible: &[bool],
    rng: &mut fastrand::Rng,
    target_s: f64,
) -> Option<(f64, f64)> {
    for _ in 0..400 {
        let i = rng.usize(1..speech.len().max(2));
        if i >= speech.len()
            || speech[i].0 - speech[i - 1].1 < 3.0
            || !audible.get(i).copied().unwrap_or(false)
        {
            continue;
        }
        let start = speech[i].0;
        let mut end = speech[i].1;
        let mut ok = true;
        for w in speech[i..].windows(2) {
            if w[1].0 - w[0].1 > 12.0 {
                ok = end - start >= 20.0;
                break;
            }
            end = w[1].1;
            if end - start >= target_s {
                break;
            }
        }
        if ok && end - start >= 20.0 {
            return Some((start, end.min(start + target_s + 15.0)));
        }
    }
    None
}

/// Cheap Schroeder reverb (4 combs + 2 allpasses): enough to make close-talk
/// read speech sound like it was picked up across a desk.
fn reverb(x: &[f32]) -> Vec<f32> {
    let combs = [405usize, 431, 463, 492];
    let allpasses = [202usize, 160];
    let mut wet = vec![0.0f32; x.len()];
    for &d in &combs {
        let mut buf = vec![0.0f32; d];
        let mut idx = 0;
        for (i, &s) in x.iter().enumerate() {
            let y = buf[idx];
            buf[idx] = s + y * 0.78;
            idx = (idx + 1) % d;
            wet[i] += y * 0.25;
        }
    }
    for &d in &allpasses {
        let mut buf = vec![0.0f32; d];
        let mut idx = 0;
        for v in wet.iter_mut() {
            let b = buf[idx];
            let y = -*v * 0.5 + b;
            buf[idx] = *v + b * 0.5;
            idx = (idx + 1) % d;
            *v = y;
        }
    }
    x.iter().zip(&wet).map(|(d, w)| 0.8 * d + 0.35 * w).collect()
}

fn gain_to(x: &[f32], target_db: f32) -> f32 {
    10f32.powf((target_db - rms_db_of(x)) / 20.0)
}

/// Loudness of the speech-active part (frames within 25 dB of the loudest).
fn active_rms_db(x: &[f32]) -> f32 {
    let frames: Vec<f32> = x.chunks(SR / 50).map(rms_db_of).collect();
    let peak = frames.iter().copied().fold(-100.0, f32::max);
    let active: Vec<f32> = frames.into_iter().filter(|&f| f > peak - 25.0).collect();
    if active.is_empty() {
        return -100.0;
    }
    let mean_power = active.iter().map(|db| 10f32.powf(db / 10.0)).sum::<f32>() / active.len() as f32;
    10.0 * mean_power.log10()
}

struct AmiMeeting {
    path: PathBuf,
    /// Merged speech segments (seconds), all speakers.
    speech: Vec<(f64, f64)>,
    /// Per segment: clearly audible at the far-field mic.
    audible: Vec<bool>,
}

struct Sources {
    ami: Vec<AmiMeeting>,
    fleurs: Vec<PathBuf>,
    noise: Vec<PathBuf>,
    music: Vec<PathBuf>,
}

fn load_sources(raw: &Path) -> Res<Sources> {
    let segdir = raw.join("ami/annotations/segments");
    let mut ami = Vec::new();
    for p in wavs_under(&raw.join("ami")) {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let Some(meeting) = name.strip_suffix(".Array1-01.wav") else { continue };
        let speech = ami_speech(&segdir, meeting);
        if !speech.is_empty() {
            let audible = audible_segments(&read_any(&p)?, &speech);
            ami.push(AmiMeeting { path: p, speech, audible });
        }
    }
    let fleurs = wavs_under(&raw.join("fleurs"));
    let noise = wavs_under(&raw.join("musan/musan/noise"));
    let music = wavs_under(&raw.join("musan/musan/music"));
    if ami.is_empty() || fleurs.is_empty() || noise.is_empty() || music.is_empty() {
        return Err(format!(
            "missing sources under {} (ami {}, fleurs {}, noise {}, music {}); run scripts/fetch-corpus.sh",
            raw.display(),
            ami.len(),
            fleurs.len(),
            noise.len(),
            music.len()
        )
        .into());
    }
    Ok(Sources { ami, fleurs, noise, music })
}

fn read_any(p: &Path) -> Res<Vec<f32>> {
    read_wav_16k(p).map_err(|e| format!("{}: {e}", p.display()).into())
}

/// One conversation: (samples, source description).
fn make_conversation(src: &Sources, rng: &mut fastrand::Rng, use_ami: bool) -> Res<(Vec<f32>, String)> {
    if use_ami {
        for _ in 0..20 {
            let AmiMeeting { path, speech, audible } = &src.ami[rng.usize(..src.ami.len())];
            let target_s = rng.f64() * 100.0 + 40.0;
            let Some((a, b)) = ami_excerpt(speech, audible, rng, target_s) else { continue };
            let audio = read_any(path)?;
            let (s, e) = ((a * SR as f64) as usize, ((b * SR as f64) as usize).min(audio.len()));
            if e > s + 20 * SR {
                let name = path.file_name().unwrap().to_string_lossy().replace(".Array1-01.wav", "");
                return Ok((audio[s..e].to_vec(), format!("ami:{name}@{a:.1}")));
            }
        }
        return Err("could not find a usable AMI excerpt".into());
    }
    // FLEURS "dialogue": alternating utterances with short turn gaps.
    let turns = rng.usize(3..=8);
    let mut out = Vec::new();
    for t in 0..turns {
        let utt = read_any(&src.fleurs[rng.usize(..src.fleurs.len())])?;
        out.extend_from_slice(&utt);
        if t + 1 < turns {
            out.extend(std::iter::repeat_n(0.0, (rng.f32() * 1.7 * SR as f32 + 0.3 * SR as f32) as usize));
        }
    }
    Ok((reverb(&out), format!("fleurs-id:{turns}turns")))
}

pub fn cmd_mix(mut args: Vec<String>) -> Res<()> {
    let raw =
        PathBuf::from(crate::take_opt(&mut args, "--raw").unwrap_or_else(|| "testdata/corpus/raw".into()));
    let out_dir =
        PathBuf::from(crate::take_opt(&mut args, "--out").unwrap_or_else(|| "testdata/corpus/real".into()));
    let streams: usize = crate::take_opt(&mut args, "--streams").map(|v| v.parse()).transpose()?.unwrap_or(8);
    let minutes: f64 =
        crate::take_opt(&mut args, "--minutes").map(|v| v.parse()).transpose()?.unwrap_or(30.0);
    let seed: u64 = crate::take_opt(&mut args, "--seed").map(|v| v.parse()).transpose()?.unwrap_or(7);
    let src = load_sources(&raw)?;
    std::fs::create_dir_all(&out_dir)?;
    let (aud, all) = src
        .ami
        .iter()
        .fold((0, 0), |(a, n), m| (a + m.audible.iter().filter(|x| **x).count(), n + m.audible.len()));
    println!("AMI: {aud}/{all} merged speech segments are clearly audible at the far-field mic");
    println!(
        "sources: {} AMI meetings, {} FLEURS-id utterances, {} MUSAN noise, {} MUSAN music",
        src.ami.len(),
        src.fleurs.len(),
        src.noise.len(),
        src.music.len()
    );
    let total = (minutes * 60.0) as usize * SR;

    for k in 0..streams {
        let mut rng = fastrand::Rng::with_seed(seed.wrapping_mul(1_000_003).wrapping_add(k as u64));
        let snr = SNRS_DB[k % SNRS_DB.len()];

        // Background: concatenated MUSAN noise, levelled to BED_DBFS.
        let mut bed = Vec::with_capacity(total);
        let mut bed_log: Vec<(usize, usize, String)> = Vec::new();
        while bed.len() < total {
            let path = &src.noise[rng.usize(..src.noise.len())];
            let n = read_any(path)?;
            if n.len() < SR {
                continue;
            }
            let g = gain_to(&n, BED_DBFS);
            let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
            bed_log.push((bed.len(), (bed.len() + n.len()).min(total), name));
            bed.extend(n.iter().map(|s| s * g));
        }
        bed.truncate(total);
        let mut mixed = bed;

        // Conversations, at least MIN_GAP_S apart.
        let mut labels: Vec<(usize, usize, String)> = Vec::new();
        let mut cursor = MIN_GAP_S * SR;
        let mut n = 0;
        while cursor < total {
            let (conv, what) = make_conversation(&src, &mut rng, n % 2 == 0)?;
            if cursor + conv.len() + MIN_GAP_S * SR > total {
                break;
            }
            let g = 10f32.powf((BED_DBFS + snr - active_rms_db(&conv)) / 20.0);
            for (i, s) in conv.iter().enumerate() {
                mixed[cursor + i] += s * g;
            }
            labels.push((cursor, cursor + conv.len(), what));
            cursor += conv.len() + (MIN_GAP_S + rng.usize(0..120)) * SR;
            n += 1;
        }

        // Hard negatives in the gaps: music clips (some with vocals), never labelled.
        let mut distractors: Vec<(usize, usize, String)> = Vec::new();
        for _ in 0..6 {
            let music_path = &src.music[rng.usize(..src.music.len())];
            let music_name = music_path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
            let music = read_any(music_path)?;
            let len = (rng.usize(10..40) * SR).min(music.len());
            if len < 5 * SR {
                continue;
            }
            let off = rng.usize(0..music.len() - len + 1);
            let clip = &music[off..off + len];
            // Free slot: >= 5 s away from every conversation.
            for _ in 0..50 {
                let at = rng.usize(0..total - len);
                let clear = labels.iter().all(|&(s, e, _)| at + len + 5 * SR < s || at > e + 5 * SR)
                    && distractors.iter().all(|&(s, e, _)| at + len < s || at > e);
                if clear {
                    let g = gain_to(clip, BED_DBFS + 6.0);
                    for (i, s) in clip.iter().enumerate() {
                        mixed[at + i] += s * g;
                    }
                    distractors.push((at, at + len, format!("music:{music_name}")));
                    break;
                }
            }
        }

        let name = format!("mix{k:02}_snr{}", snr as i32);
        let wav = out_dir.join(format!("{name}.wav"));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&wav, spec)?;
        for s in &mixed {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
        let secs = |i: usize| i as f64 / SR as f64;
        let mut l = String::from("# start_sec,end_sec,source  (conversations that must be recorded)\n");
        for (s, e, what) in &labels {
            l.push_str(&format!("{:.3},{:.3},{what}\n", secs(*s), secs(*e)));
        }
        std::fs::write(labels_path(&wav), l)?;
        let mut d = String::from("# start_sec,end_sec,kind  (unlabelled hard negatives)\n");
        for (s, e, what) in &distractors {
            d.push_str(&format!("{:.3},{:.3},{what}\n", secs(*s), secs(*e)));
        }
        let mut dp = wav.as_os_str().to_owned();
        dp.push(".distractors.csv");
        std::fs::write(PathBuf::from(dp), d)?;
        let mut b = String::from("# start_sec,end_sec,musan_noise_file  (background bed)\n");
        for (s, e, what) in &bed_log {
            b.push_str(&format!("{:.3},{:.3},{what}\n", secs(*s), secs(*e)));
        }
        let mut bp = wav.as_os_str().to_owned();
        bp.push(".bed.csv");
        std::fs::write(PathBuf::from(bp), b)?;
        println!(
            "wrote {} ({minutes:.0} min, SNR {snr} dB, {} conversations, {} music clips)",
            wav.display(),
            labels.len(),
            distractors.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ami_excerpts_start_after_quiet_and_never_span_long_pauses() {
        // 5 s quiet, then speech with short gaps, a 20 s pause, more speech.
        let speech = vec![(0.0, 1.0), (6.0, 10.0), (11.0, 20.0), (21.0, 35.0), (55.0, 70.0), (71.0, 90.0)];
        let audible = vec![true; speech.len()];
        let mut rng = fastrand::Rng::with_seed(1);
        for _ in 0..100 {
            let Some((a, b)) = ami_excerpt(&speech, &audible, &mut rng, 60.0) else { continue };
            // Starts at a segment start preceded by >= 3 s of quiet.
            let i = speech.iter().position(|s| s.0 == a).expect("starts at a segment");
            assert!(i > 0 && speech[i].0 - speech[i - 1].1 >= 3.0, "{a}");
            // Never crosses the 20 s pause (35 -> 55).
            assert!(!(a < 35.0 && b > 55.0), "{a}..{b} spans the long pause");
            assert!(b - a >= 20.0);
        }
    }

    #[test]
    fn inaudible_segments_are_never_excerpt_starts() {
        let speech = vec![(0.0, 1.0), (6.0, 10.0), (14.0, 40.0), (44.0, 80.0)];
        let audible = vec![true, false, true, false];
        let mut rng = fastrand::Rng::with_seed(3);
        for _ in 0..100 {
            if let Some((a, _)) = ami_excerpt(&speech, &audible, &mut rng, 60.0) {
                assert_eq!(a, 14.0, "excerpt started at an inaudible segment");
            }
        }
    }

    #[test]
    fn audibility_compares_segment_level_with_local_floor() {
        // 60 s of quiet room noise with a loud 2 s burst at 20 s and a faint one at 40 s.
        let mut rng = fastrand::Rng::with_seed(9);
        let mut audio: Vec<f32> = (0..60 * SR).map(|_| (rng.f32() - 0.5) * 0.002).collect();
        // Noise floor is about -65 dBFS; the faint burst lands only ~3 dB above it.
        for (start, amp) in [(20, 0.05f32), (40, 0.0008)] {
            for (i, v) in audio.iter_mut().enumerate().skip(start * SR).take(2 * SR) {
                *v += amp * (i as f32 * 0.07).sin();
            }
        }
        assert_eq!(audible_segments(&audio, &[(20.0, 22.0), (40.0, 42.0)]), vec![true, false]);
    }

    #[test]
    fn reverb_keeps_length_and_adds_a_tail_like_energy() {
        let mut x = vec![0.0f32; SR];
        x[100] = 1.0;
        let y = reverb(&x);
        assert_eq!(y.len(), x.len());
        let late: f32 = y[SR / 10..SR / 2].iter().map(|v| v.abs()).sum();
        assert!(late > 0.01, "no reverberant tail");
        assert!(y.iter().all(|v| v.is_finite() && v.abs() < 2.0));
    }

    #[test]
    fn active_loudness_ignores_silence() {
        let mut x: Vec<f32> = (0..SR).map(|i| 0.1 * (i as f32 * 0.05).sin()).collect();
        let speech_db = active_rms_db(&x);
        x.extend(std::iter::repeat_n(0.0, 4 * SR)); // 80 % silence
        assert!((active_rms_db(&x) - speech_db).abs() < 0.5, "silence diluted the loudness");
    }

    #[test]
    fn merges_overlapping_speaker_segments() {
        let dir = std::env::temp_dir().join(format!("mix-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let seg = |s: f64, e: f64| {
            format!("<segment nite:id=\"x\" transcriber_start=\"{s}\" transcriber_end=\"{e}\">\n")
        };
        std::fs::write(dir.join("M1.A.segments.xml"), seg(1.0, 5.0) + &seg(10.0, 12.0)).unwrap();
        std::fs::write(dir.join("M1.B.segments.xml"), seg(4.0, 8.0)).unwrap();
        assert_eq!(ami_speech(&dir, "M1"), vec![(1.0, 8.0), (10.0, 12.0)]);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
