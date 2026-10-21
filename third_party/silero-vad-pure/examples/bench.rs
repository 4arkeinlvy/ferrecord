//! Per-chunk inference latency and real-time factor, 16k and 8k.
//! Run: cargo run --release --example bench

use std::time::Instant;

use silero_vad_pure::{SampleRate, SileroVad};

fn bench(rate: SampleRate, iters: usize) {
    let mut vad = SileroVad::new(rate).unwrap();
    let chunk_size = vad.chunk_size();
    let sr = match rate {
        SampleRate::Hz16000 => 16000,
        SampleRate::Hz8000 => 8000,
    };
    // pseudo-random input, deterministic
    let mut seed = 0x12345678u32;
    let chunk: Vec<f32> = (0..chunk_size)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
        })
        .collect();

    // warmup
    for _ in 0..50 {
        std::hint::black_box(vad.process(std::hint::black_box(&chunk)).unwrap());
    }

    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(vad.process(std::hint::black_box(&chunk)).unwrap());
    }
    let elapsed = start.elapsed();
    let per_chunk = elapsed / iters as u32;
    let chunk_audio_secs = chunk_size as f64 / sr as f64;
    let rtf = chunk_audio_secs / per_chunk.as_secs_f64();
    println!(
        "{sr} Hz: {per_chunk:?}/chunk ({chunk_size} samples = {:.0} ms audio) -> {rtf:.0}x realtime",
        chunk_audio_secs * 1000.0
    );
}

fn main() {
    bench(SampleRate::Hz16000, 2000);
    bench(SampleRate::Hz8000, 2000);
}
