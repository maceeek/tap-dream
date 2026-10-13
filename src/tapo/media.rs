//! The camera's media server (TCP port 8800): digest-authenticated
//! multipart/mixed stream with AES-CBC encrypted payloads. Port of pytapo's
//! `media_stream/session.py`, `crypto.py` and `downloader.py`.

use super::crypto::{aes_cbc_decrypt, md5, md5_hex_upper, random_hex, sha256_hex_upper};
use super::https::find;
use super::pyjson;
use super::ts::TsDuration;
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

const CLIENT_BOUNDARY: &str = "--client-stream-boundary--";
const USERNAME: &str = "admin";

pub struct Part {
    pub mimetype: String,
    pub body: Vec<u8>,
    pub json: Option<Value>,
}

pub struct MediaSession {
    sock: TcpStream,
    buf: Vec<u8>,
    device_boundary: Vec<u8>,
    key: [u8; 16],
    iv: [u8; 16],
    window: u32,
}

fn parse_headers(block: &[u8]) -> HashMap<String, String> {
    String::from_utf8_lossy(block)
        .trim()
        .split("\r\n")
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect()
}

/// Some devices answer "HTTP ERROR 401HTTP/1.0 200 OK"; take the last status line.
fn parse_status(line: &[u8]) -> Result<u16> {
    let s = String::from_utf8_lossy(line);
    let start = s.find("HTTP/").ok_or_else(|| anyhow!("bad HTTP status line {s:?}"))?;
    let s = &s[start..];
    let s = match s[5..].find("HTTP/") {
        Some(p) => &s[5 + p..],
        None => s,
    };
    s.split_whitespace().nth(1).and_then(|c| c.parse().ok()).ok_or_else(|| anyhow!("bad HTTP status line {s:?}"))
}

fn unquote(s: &str) -> String {
    s.trim().replace('"', "")
}

/// `WWW-Authenticate: Digest realm="..",nonce="..",...` -> map
pub fn parse_digest_challenge(h: &str) -> HashMap<String, String> {
    let rest = h.split_once(' ').map(|(_, r)| r).unwrap_or("");
    rest.split(',').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (unquote(k), unquote(v))).collect()
}

/// `Key-Exchange: username="admin" nonce="..."` -> map
pub fn parse_key_exchange(h: &str) -> HashMap<String, String> {
    h.split(' ').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (unquote(k), unquote(v))).collect()
}

/// Port 8800 advertises encrypt_type="3" when it expects the SHA-256 password hash.
pub fn media_password_hash(challenge: &HashMap<String, String>, cloud_password: &str) -> String {
    if challenge.get("encrypt_type").map(String::as_str) == Some("3") {
        sha256_hex_upper(cloud_password)
    } else {
        md5_hex_upper(cloud_password)
    }
}

pub fn digest_response(challenge: &HashMap<String, String>, hashed_password: &str, cnonce: &str) -> String {
    let realm = challenge.get("realm").map(String::as_str).unwrap_or("");
    let nonce = challenge.get("nonce").map(String::as_str).unwrap_or("");
    let c1 = hex::encode(md5(format!("{USERNAME}:{realm}:{hashed_password}").as_bytes()));
    let c2 = hex::encode(md5(b"POST:/stream"));
    hex::encode(md5(format!("{c1}:{nonce}:00000001:{cnonce}:auth:{c2}").as_bytes()))
}

/// AES key/iv for a key exchange (see pytapo AESHelper).
pub fn session_keys(kx: &HashMap<String, String>, hashed_password: &str) -> Result<([u8; 16], [u8; 16])> {
    let nonce = kx.get("nonce").ok_or_else(|| anyhow!("nonce is missing from key exchange"))?;
    let username = kx.get("username").map(String::as_str).unwrap_or("");
    if username == "none" {
        bail!("media encryption is turned off on the camera; this mode needs the device's private key and is not supported");
    }
    let key = md5(format!("{nonce}:{hashed_password}").as_bytes());
    let iv = md5(format!("{username}:{nonce}").as_bytes());
    Ok((key, iv))
}

impl MediaSession {
    pub fn open(host: &str, port: u16, cloud_password: &str, window: u32, timeout: Duration) -> Result<Self> {
        let addr = (host, port).to_socket_addrs()?.next().ok_or_else(|| anyhow!("cannot resolve {host}"))?;
        let sock =
            TcpStream::connect_timeout(&addr, timeout).with_context(|| format!("cannot connect to the media server {host}:{port}"))?;
        sock.set_read_timeout(Some(timeout))?;
        sock.set_write_timeout(Some(timeout))?;
        sock.set_nodelay(true)?;
        let mut s =
            Self { sock, buf: Vec::new(), device_boundary: b"--device-stream-boundary--".to_vec(), key: [0; 16], iv: [0; 16], window };

        let req_line = "POST /stream HTTP/1.1";
        let mut headers: Vec<(&str, String)> = vec![
            ("Content-Type", format!("multipart/mixed;boundary={CLIENT_BOUNDARY}")),
            ("Connection", "keep-alive".into()),
            ("Content-Length", "-1".into()),
        ];
        // Step one: unauthenticated request to get the digest challenge.
        s.send_http_request(req_line, &headers)?;
        let data = s.read_until(b"\r\n\r\n")?;
        let (line, rest) = data.split_at(find(&data, b"\r\n").unwrap_or(data.len()));
        let _ = parse_status(line)?;
        let h = parse_headers(rest);
        let www = h.get("WWW-Authenticate").ok_or_else(|| anyhow!("media server sent no digest challenge"))?;
        let challenge = parse_digest_challenge(www);
        let hashed = media_password_hash(&challenge, cloud_password);
        let cnonce = random_hex(24);
        let response = digest_response(&challenge, &hashed, &cnonce);
        let g = |k: &str| challenge.get(k).cloned().unwrap_or_default();
        let auth = format!(
            "Digest username=\"{USERNAME}\",realm=\"{}\",uri=\"/stream\",algorithm=MD5,nonce=\"{}\",nc=00000001,cnonce=\"{cnonce}\",qop=auth,response=\"{response}\",opaque=\"{}\"",
            g("realm"),
            g("nonce"),
            g("opaque")
        );
        headers.push(("Authorization", auth));

        // Step two: the authenticated request opens the stream.
        s.send_http_request(req_line, &headers)?;
        let data = s.read_until(b"\r\n\r\n")?;
        let (line, rest) = data.split_at(find(&data, b"\r\n").unwrap_or(data.len()));
        let status = parse_status(line)?;
        if status != 200 {
            if status == 401 {
                bail!("media server rejected the cloud password (HTTP 401)");
            }
            bail!("media server returned HTTP {status}");
        }
        let h = parse_headers(rest);
        let kx = h.get("Key-Exchange").ok_or_else(|| anyhow!("server reply does not contain the required Key-Exchange"))?;
        if let Some(ct) = h.get("Content-Type") {
            if let Some(b) = ct.split(';').find_map(|c| c.trim().strip_prefix("boundary=")) {
                s.device_boundary = b.as_bytes().to_vec();
            }
        }
        let (key, iv) = session_keys(&parse_key_exchange(kx), &hashed)?;
        s.key = key;
        s.iv = iv;
        Ok(s)
    }

    pub fn set_read_timeout(&self, t: Duration) -> Result<()> {
        self.sock.set_read_timeout(Some(t))?;
        Ok(())
    }

    fn send_http_request(&mut self, delimiter: &str, headers: &[(&str, String)]) -> Result<()> {
        let mut out = format!("{delimiter}\r\n");
        for (k, v) in headers {
            out.push_str(&format!("{k}: {v}\r\n"));
        }
        out.push_str("\r\n");
        self.sock.write_all(out.as_bytes())?;
        Ok(())
    }

    fn fill(&mut self) -> Result<()> {
        let mut tmp = [0u8; 64 * 1024];
        let n = self.sock.read(&mut tmp)?;
        if n == 0 {
            bail!(std::io::Error::new(ErrorKind::UnexpectedEof, "media server closed the connection"));
        }
        self.buf.extend_from_slice(&tmp[..n]);
        Ok(())
    }

    fn read_until(&mut self, delim: &[u8]) -> Result<Vec<u8>> {
        let mut from = 0;
        loop {
            if let Some(p) = find(&self.buf[from..], delim) {
                let end = from + p + delim.len();
                return Ok(self.buf.drain(..end).collect());
            }
            from = self.buf.len().saturating_sub(delim.len());
            self.fill()?;
        }
    }

    fn read_exact_n(&mut self, n: usize) -> Result<Vec<u8>> {
        while self.buf.len() < n {
            self.fill()?;
        }
        Ok(self.buf.drain(..n).collect())
    }

    /// Sends a JSON request part (`transceive`); returns the sequence number used.
    pub fn send_request(&mut self, mut request: Value) -> Result<i64> {
        // A random high sequence number, avoiding multiples of the window
        // (those trigger acknowledgements).
        let mut seq;
        loop {
            seq = rand::Rng::gen_range(&mut rand::thread_rng(), 1000..0x7FFF);
            if seq % self.window as i64 != 0 {
                break;
            }
        }
        request["seq"] = json!(seq);
        let data = pyjson::dumps_compact(&request);
        let headers = vec![
            ("Content-Type", "application/json".to_string()),
            ("Content-Length", data.len().to_string()),
            ("X-Data-Window-Size", self.window.to_string()),
        ];
        self.send_http_request(&format!("--{CLIENT_BOUNDARY}"), &headers)?;
        self.sock.write_all(data.as_bytes())?;
        self.sock.write_all(b"\r\n")?;
        Ok(seq)
    }

    fn acknowledge(&mut self, session: Option<i64>, seq: i64) -> Result<()> {
        let data = r#"{"type":"notification","params":{"event_type":"stream_sequence"}}"#;
        let w = self.window as i64;
        let headers = vec![
            ("X-Session-Id", session.map(|s| s.to_string()).unwrap_or_else(|| "None".into())),
            ("X-Data-Received", (w * (seq / w)).to_string()),
            ("Content-Length", data.len().to_string()),
        ];
        self.send_http_request(&format!("--{CLIENT_BOUNDARY}"), &headers)?;
        self.sock.write_all(data.as_bytes())?;
        Ok(())
    }

    /// Reads the next part of the multipart stream, decrypting it if needed.
    pub fn next_part(&mut self) -> Result<Part> {
        let boundary = self.device_boundary.clone();
        self.read_until(&boundary)?;
        let block = self.read_until(b"\r\n\r\n")?;
        let h = parse_headers(&block);
        let mimetype = h.get("Content-Type").cloned().unwrap_or_default();
        let length: usize =
            h.get("Content-Length").and_then(|v| v.parse().ok()).ok_or_else(|| anyhow!("stream part without Content-Length"))?;
        let encrypted = h.get("X-If-Encrypt").map(|v| v.trim() == "1").unwrap_or(false);
        let mut session = h.get("X-Session-Id").and_then(|v| v.parse().ok());
        let mut seq = h.get("X-Data-Sequence").and_then(|v| v.parse().ok());
        let raw = self.read_exact_n(length)?;
        let body = if encrypted {
            aes_cbc_decrypt(&self.key, &self.iv, &raw)
                .map_err(|_| anyhow!("cannot decrypt the stream - this usually means that the cloud password is incorrect"))?
        } else {
            raw
        };
        let mut json = None;
        if mimetype == "application/json" {
            if let Ok(v) = serde_json::from_slice::<Value>(&body) {
                if let Some(s) = v.get("seq").and_then(Value::as_i64) {
                    seq = Some(s);
                }
                if let Some(s) = v.pointer("/params/session_id") {
                    session = s.as_i64().or_else(|| s.as_str().and_then(|x| x.parse().ok()));
                }
                json = Some(v);
            }
        }
        if let Some(s) = seq {
            if s % self.window as i64 == 0 {
                self.acknowledge(session, s)?;
            }
        }
        Ok(Part { mimetype, body, json })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Download,
    Playback,
}

pub struct DownloadRequest<'a> {
    pub host: &'a str,
    pub port: u16,
    pub cloud_password: &'a str,
    pub user_id: Value,
    pub player_id: &'a str,
    pub start: i64,
    pub end: i64,
    pub window: u32,
    pub stall_timeout: Duration,
}

#[derive(Debug)]
pub struct DownloadResult {
    pub bytes: u64,
    pub seconds: f64,
    pub finished: bool,
    pub method: Method,
}

fn build_request(req: &DownloadRequest, method: Method) -> Value {
    match method {
        Method::Download => json!({"type": "request", "seq": 1, "params": {
            "download": {
                "client_id": req.user_id,
                "channels": [0],
                "media_type": 0,
                "start_time": req.start.to_string(),
                "end_time": req.end.to_string(),
                "player_id": req.player_id,
            },
            "method": "get"
        }}),
        Method::Playback => json!({"type": "request", "seq": 1, "params": {
            "playback": {
                "client_id": req.user_id,
                "channels": [0, 1],
                "scale": "1/1",
                "start_time": req.start.to_string(),
                "end_time": req.end.to_string(),
                "event_type": [1, 2],
            },
            "method": "get"
        }}),
    }
}

pub enum Attempt {
    Done(DownloadResult),
    /// The firmware does not know the fast download request.
    Unsupported,
}

/// One streaming attempt; MPEG-TS goes to `out`.
pub fn stream_recording(req: &DownloadRequest, method: Method, out: &mut dyn Write, progress: &mut dyn FnMut(f64)) -> Result<Attempt> {
    const PADDING: f64 = 5.0;
    let length = (req.end - req.start) as f64;
    let mut s = MediaSession::open(req.host, req.port, req.cloud_password, req.window, super::CONNECTION_TIMEOUT)?;
    s.send_request(build_request(req, method))?;
    s.set_read_timeout(req.stall_timeout)?;
    let mut ts = TsDuration::new();
    let mut bytes = 0u64;
    let mut finished = false;
    loop {
        let part = match s.next_part() {
            Ok(p) => p,
            Err(e) => {
                let stalled = e
                    .downcast_ref::<std::io::Error>()
                    .map(|io| matches!(io.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::UnexpectedEof))
                    .unwrap_or(false);
                if stalled && bytes > 0 {
                    log::debug!("stream stopped: {e}");
                    break;
                }
                return Err(e);
            }
        };
        match part.mimetype.as_str() {
            "video/mp2t" => {
                out.write_all(&part.body)?;
                bytes += part.body.len() as u64;
                ts.feed(&part.body);
                progress(ts.seconds());
                if ts.seconds() > length + PADDING {
                    finished = true;
                    break;
                }
            }
            "application/json" => {
                let Some(j) = part.json else { continue };
                let kind = j.get("type").and_then(Value::as_str).unwrap_or("");
                let code = j.pointer("/params/error_code").and_then(Value::as_i64).unwrap_or(0);
                if kind == "response" && code != 0 {
                    if method == Method::Download {
                        log::debug!("camera refused the download request ({code}), falling back to playback");
                        return Ok(Attempt::Unsupported);
                    }
                    bail!("camera refused the playback request: error {code} ({})", super::error_message(code));
                }
                if kind == "notification"
                    && j.pointer("/params/event_type").and_then(Value::as_str) == Some("stream_status")
                    && j.pointer("/params/status").and_then(Value::as_str) == Some("finished")
                {
                    finished = true;
                    break;
                }
            }
            _ => {}
        }
    }
    out.flush()?;
    Ok(Attempt::Done(DownloadResult { bytes, seconds: ts.seconds(), finished, method }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_and_keys_match_pytapo() {
        let v: Value = serde_json::from_str(include_str!("testdata/media.json")).unwrap();
        let challenge = parse_digest_challenge(v["www_authenticate"].as_str().unwrap());
        let hashed = media_password_hash(&challenge, v["password"].as_str().unwrap());
        assert_eq!(hashed, v["hashed"].as_str().unwrap());
        assert_eq!(digest_response(&challenge, &hashed, v["cnonce"].as_str().unwrap()), v["response"].as_str().unwrap());
        let (key, iv) = session_keys(&parse_key_exchange(v["key_exchange"].as_str().unwrap()), &hashed).unwrap();
        assert_eq!(hex::encode(key), v["key"].as_str().unwrap());
        assert_eq!(hex::encode(iv), v["iv"].as_str().unwrap());
        assert_eq!(parse_status(b"HTTP ERROR 401HTTP/1.0 200 OK").unwrap(), 200);
    }
}
