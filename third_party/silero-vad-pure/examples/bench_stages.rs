//! Per-stage latency breakdown (stft / enc0..3 / lstm+head), 16k and 8k.
//! Run: cargo run --release --features profile --example bench_stages

use silero_vad_pure::SampleRate;

#[cfg(feature = "profile")]
mod imp {
    use std::time::Duration;

    use silero_vad_pure::{SampleRate, SileroVad};

    pub fn run(rate: SampleRate, iters: usize) {
        let mut vad = SileroVad::new(rate).unwrap();
        let chunk_size = vad.chunk_size();
        let sr = match rate {
            SampleRate::Hz16000 => 16000,
            SampleRate::Hz8000 => 8000,
        };
        let mut seed = 0x12345678u32;
        let chunk: Vec<f32> = (0..chunk_size)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
            })
            .collect();

        for _ in 0..50 {
            std::hint::black_box(vad.process(std::hint::black_box(&chunk)).unwrap());
        }
        vad.reset_profile();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(vad.process(std::hint::black_box(&chunk)).unwrap());
        }
        let total = start.elapsed();
        let p = vad.stage_profile();
        let n = p.chunks as f64;
        let total_ns = total.as_nanos() as f64 / n;
        let stages: [(&str, Duration); 6] = [
            ("stft", p.stft),
            ("enc0", p.enc[0]),
            ("enc1", p.enc[1]),
            ("enc2", p.enc[2]),
            ("enc3", p.enc[3]),
            ("lstm+head", p.lstm_head),
        ];
        println!("{sr} Hz: {total_ns:.0} ns/chunk total (incl. ~profile overhead)");
        for (name, d) in stages {
            let ns = d.as_nanos() as f64 / n;
            println!("  {name:<9} {ns:>7.0} ns  {:>5.1}%", 100.0 * ns / total_ns);
        }
    }
}

fn main() {
    #[cfg(feature = "profile")]
    {
        imp::run(SampleRate::Hz16000, 2000);
        imp::run(SampleRate::Hz8000, 2000);
    }
    #[cfg(not(feature = "profile"))]
    {
        let _ = SampleRate::Hz16000;
        eprintln!("bench_stages requires `--features profile`");
    }
}
