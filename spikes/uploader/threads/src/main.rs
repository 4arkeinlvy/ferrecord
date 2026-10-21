//! Variant A: OS threads + blocking ureq (the current design).
use std::collections::VecDeque;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use spike_common::*;
#[cfg(feature = "tls")]
use ureq::tls::{TlsConfig, TlsProvider};
use ureq::{Agent, SendBody};

fn agent(send_body_timeout: Duration) -> Agent {
    let builder = Agent::config_builder();
    #[cfg(feature = "tls")]
    let builder = builder.tls_config(TlsConfig::builder().provider(TlsProvider::NativeTls).build());
    builder
        .timeout_connect(Some(Duration::from_secs(10)))
        .timeout_send_body(Some(send_body_timeout))
        .timeout_recv_response(Some(Duration::from_secs(30)))
        .http_status_as_error(false)
        .build()
        .into()
}

/// Checks the stop flag on every body read: the best a blocking client can do.
struct Stoppable<R> {
    inner: R,
    stop: Arc<AtomicBool>,
}

impl<R: Read> Read for Stoppable<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.stop.load(Ordering::Relaxed) {
            return Err(std::io::Error::other("stopped"));
        }
        self.inner.read(buf)
    }
}

fn put_part(agent: &Agent, url: &str, path: &Path, off: u64, len: u64, stop: &Arc<AtomicBool>) -> Result<String, String> {
    let mut body = Stoppable { inner: PartReader::open(path, off, len), stop: Arc::clone(stop) };
    let resp = agent
        .put(url)
        .header("Content-Length", len.to_string())
        .send(SendBody::from_reader(&mut body))
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("status {}", resp.status()));
    }
    Ok(resp.headers().get("etag").and_then(|v| v.to_str().ok()).unwrap_or_default().to_owned())
}

fn main() {
    let endpoint = std::env::args().nth(1).unwrap_or_else(|| "http://127.0.0.1:9000".into());
    let path = data_file();
    let agent = agent(Duration::from_secs(300));
    let idle = proc_status();
    let target = Target::new(&endpoint, "spike", "threads.bin");
    let never = Arc::new(AtomicBool::new(false));

    let _ = agent.put(&target.create_bucket_url()).send_empty();
    let t0 = Instant::now();
    let mut resp = agent.post(&target.create_upload_url()).send_empty().expect("create upload");
    let upload_id = Target::parse_upload_id(&resp.body_mut().read_to_string().expect("xml"));

    let jobs = Mutex::new(parts().into_iter().collect::<VecDeque<_>>());
    let etags = Mutex::new(vec![String::new(); parts().len()]);
    let mid_threads = AtomicU64::new(0);
    std::thread::scope(|s| {
        for _ in 0..CONCURRENCY {
            s.spawn(|| {
                loop {
                    let Some((n, off, len)) = jobs.lock().unwrap().pop_front() else { break };
                    let etag = put_part(&agent, &target.part_url(&upload_id, n), &path, off, len, &never).expect("part");
                    etags.lock().unwrap()[usize::from(n) - 1] = etag;
                    let _ = mid_threads.compare_exchange(0, proc_status().threads, Ordering::Relaxed, Ordering::Relaxed);
                }
            });
        }
    });
    let (url, xml) = target.complete(&upload_id, &etags.lock().unwrap());
    let resp = agent.post(&url).send(xml.as_str()).expect("complete");
    assert!(resp.status().is_success(), "complete: {}", resp.status());
    let upload = t0.elapsed();
    let end = proc_status();

    // Stop latency against a server that accepts and never reads.
    let hole = blackhole();
    let short = Arc::new(agent_for_stop());
    let stop = Arc::new(AtomicBool::new(false));
    let (a, s2, p) = (Arc::clone(&short), Arc::clone(&stop), path.clone());
    let started = Instant::now();
    let worker = std::thread::spawn(move || put_part(&a, &format!("http://{hole}/spike/x"), &p, 0, PART, &s2));
    std::thread::sleep(Duration::from_millis(500));
    let asked = Instant::now();
    stop.store(true, Ordering::Relaxed);
    let _ = worker.join();
    let _ = started;
    report(if cfg!(feature = "tls") { "A threads+ureq" } else { "A threads+ureq (no TLS)" }, idle, mid_threads.load(Ordering::Relaxed), end, upload, asked.elapsed());
}

/// Same agent, with a 10 s body timeout so the hung-server test finishes quickly
/// (production uses 300 s; the stop latency scales with it).
fn agent_for_stop() -> Agent {
    agent(Duration::from_secs(10))
}
