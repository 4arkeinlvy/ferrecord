//! Shared parts of the ADR 0001 spike: identical presigning, data and
//! measurement for both variants, so only the HTTP stack differs.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusty_s3::actions::{CreateMultipartUpload, S3Action};
use rusty_s3::{Bucket, Credentials, UrlStyle};

pub const PART: u64 = 16 * 1024 * 1024;
pub const TOTAL: u64 = 100 * 1000 * 1000;
pub const CONCURRENCY: usize = 2;
const EXPIRY: Duration = Duration::from_secs(1200); // backend's s3_presigned_multipart_url_expiry

pub struct Target {
    bucket: Bucket,
    creds: Credentials,
    pub key: String,
}

impl Target {
    pub fn new(endpoint: &str, bucket: &str, key: &str) -> Self {
        let url = endpoint.parse().expect("endpoint url");
        Self {
            bucket: Bucket::new(url, UrlStyle::Path, bucket.to_owned(), "us-east-1").expect("bucket"),
            creds: Credentials::new("rustfsadmin", "rustfsadmin"),
            key: key.to_owned(),
        }
    }

    pub fn create_bucket_url(&self) -> String {
        self.bucket.create_bucket(&self.creds).sign(EXPIRY).to_string()
    }

    /// POST; response XML carries the upload id.
    pub fn create_upload_url(&self) -> String {
        self.bucket.create_multipart_upload(Some(&self.creds), &self.key).sign(EXPIRY).to_string()
    }

    pub fn parse_upload_id(xml: &str) -> String {
        CreateMultipartUpload::parse_response(xml).expect("CreateMultipartUpload XML").upload_id().to_owned()
    }

    /// PUT, 1-based part number.
    pub fn part_url(&self, upload_id: &str, part: u16) -> String {
        self.bucket.upload_part(Some(&self.creds), &self.key, part, upload_id).sign(EXPIRY).to_string()
    }

    /// POST url + XML body.
    pub fn complete(&self, upload_id: &str, etags: &[String]) -> (String, String) {
        let action =
            self.bucket.complete_multipart_upload(Some(&self.creds), &self.key, upload_id, etags.iter().map(String::as_str));
        let url = action.sign(EXPIRY).to_string();
        (url, action.body())
    }
}

pub fn parts() -> Vec<(u16, u64, u64)> {
    let mut v = Vec::new();
    let mut off = 0;
    let mut n = 1;
    while off < TOTAL {
        let len = PART.min(TOTAL - off);
        v.push((n, off, len));
        off += len;
        n += 1;
    }
    v
}

/// Deterministic incompressible test data, created once.
pub fn data_file() -> PathBuf {
    let path = std::env::temp_dir().join("adr0001-spike-100MB.bin");
    if std::fs::metadata(&path).map(|m| m.len() == TOTAL).unwrap_or(false) {
        return path;
    }
    let mut w = BufWriter::new(File::create(&path).expect("create data"));
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut written = 0u64;
    let mut block = [0u8; 8192];
    while written < TOTAL {
        for chunk in block.chunks_exact_mut(8) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            chunk.copy_from_slice(&x.to_le_bytes());
        }
        let n = (TOTAL - written).min(block.len() as u64) as usize;
        w.write_all(&block[..n]).unwrap();
        written += n as u64;
    }
    w.flush().unwrap();
    path
}

/// A `Read` over one part of the file (what the real uploader does, minus decryption).
pub struct PartReader {
    file: File,
    left: u64,
}

impl PartReader {
    pub fn open(path: &Path, offset: u64, len: u64) -> Self {
        use std::io::{Seek, SeekFrom};
        let mut file = File::open(path).expect("open data");
        file.seek(SeekFrom::Start(offset)).unwrap();
        Self { file, left: len }
    }
}

impl Read for PartReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.left == 0 {
            return Ok(0);
        }
        let max = buf.len().min(self.left as usize).min(64 * 1024);
        let n = self.file.read(&mut buf[..max])?;
        self.left -= n as u64;
        Ok(n)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProcStatus {
    pub rss_kb: u64,
    pub hwm_kb: u64,
    pub threads: u64,
}

pub fn proc_status() -> ProcStatus {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        s.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    ProcStatus { rss_kb: field("VmRSS:"), hwm_kb: field("VmHWM:"), threads: field("Threads:") }
}

/// Accepts TCP connections and never reads or answers: a hung server.
pub fn blackhole() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let mut held = Vec::new();
        for s in listener.incoming().flatten() {
            held.push(s); // keep open, never read
        }
    });
    addr
}

pub fn report(variant: &str, idle: ProcStatus, mid_threads: u64, end: ProcStatus, upload: Duration, stop: Duration) {
    let exe = std::env::current_exe().ok().and_then(|p| std::fs::metadata(p).ok()).map_or(0, |m| m.len());
    println!(
        "{{\"variant\":\"{variant}\",\"exe_bytes\":{exe},\"idle_rss_kb\":{},\"idle_threads\":{},\"upload_threads\":{mid_threads},\
\"peak_rss_kb\":{},\"upload_secs\":{:.2},\"mb_per_s\":{:.1},\"stop_latency_ms\":{}}}",
        idle.rss_kb,
        idle.threads,
        end.hwm_kb,
        upload.as_secs_f64(),
        TOTAL as f64 / 1e6 / upload.as_secs_f64(),
        stop.as_millis()
    );
}
