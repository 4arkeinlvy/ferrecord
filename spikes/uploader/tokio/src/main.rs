//! Variant B: Tokio current_thread runtime + reqwest (the hybrid candidate).
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use futures_util::{StreamExt, stream};
use reqwest::header::CONTENT_LENGTH;
use spike_common::*;
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio_util::io::ReaderStream;
use tokio_util::sync::CancellationToken;

async fn put_part(client: &reqwest::Client, url: &str, path: &Path, off: u64, len: u64) -> Result<String, String> {
    let mut file = tokio::fs::File::open(path).await.map_err(|e| e.to_string())?;
    file.seek(std::io::SeekFrom::Start(off)).await.map_err(|e| e.to_string())?;
    let body = reqwest::Body::wrap_stream(ReaderStream::with_capacity(file.take(len), 64 * 1024));
    let resp = client.put(url).header(CONTENT_LENGTH, len).body(body).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("status {}", resp.status()));
    }
    Ok(resp.headers().get("etag").and_then(|v| v.to_str().ok()).unwrap_or_default().to_owned())
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let endpoint = std::env::args().nth(1).unwrap_or_else(|| "http://127.0.0.1:9000".into());
    let path: PathBuf = data_file();
    let client = reqwest::Client::builder().connect_timeout(Duration::from_secs(10)).build().expect("client");
    let idle = proc_status();
    let target = Target::new(&endpoint, "spike", "tokio.bin");

    let _ = client.put(target.create_bucket_url()).send().await;
    let t0 = Instant::now();
    let xml = client.post(target.create_upload_url()).send().await.expect("create upload").text().await.expect("xml");
    let upload_id = Target::parse_upload_id(&xml);

    let mut mid_threads = 0;
    let mut etags = vec![String::new(); parts().len()];
    let mut results = stream::iter(parts())
        .map(|(n, off, len)| {
            let (client, target, path, upload_id) = (&client, &target, &path, &upload_id);
            async move {
                let url = target.part_url(upload_id, n);
                let etag = tokio::time::timeout(Duration::from_secs(300), put_part(client, &url, path, off, len))
                    .await
                    .expect("part timeout")
                    .expect("part");
                (n, etag)
            }
        })
        .buffer_unordered(CONCURRENCY);
    while let Some((n, etag)) = results.next().await {
        etags[usize::from(n) - 1] = etag;
        if mid_threads == 0 {
            mid_threads = proc_status().threads;
        }
    }
    let (url, xml) = target.complete(&upload_id, &etags);
    let resp = client.post(url).body(xml).send().await.expect("complete");
    assert!(resp.status().is_success(), "complete: {}", resp.status());
    let upload = t0.elapsed();
    let end = proc_status();

    // Stop latency against a server that accepts and never reads.
    let hole = blackhole();
    let token = CancellationToken::new();
    let child = token.clone();
    let (c, p) = (client.clone(), path.clone());
    let task = tokio::spawn(async move {
        let url = format!("http://{hole}/spike/x");
        tokio::select! {
            r = put_part(&c, &url, &p, 0, PART) => r.map(|_| ()),
            () = child.cancelled() => Err("stopped".into()),
        }
    });
    tokio::time::sleep(Duration::from_millis(500)).await;
    let asked = Instant::now();
    token.cancel();
    let _ = task.await;
    report(if cfg!(feature = "tls") { "B tokio(current_thread)+reqwest" } else { "B tokio+reqwest (no TLS)" }, idle, mid_threads, end, upload, asked.elapsed());
}
