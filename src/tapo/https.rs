//! A deliberately small HTTPS/1.1 client for the camera's control port.
//!
//! Tapo cameras use self-signed certificates and, depending on firmware, TLS
//! settings that modern defaults refuse. pytapo therefore runs with
//! `ALL:@SECLEVEL=0` and no verification; we do the same with OpenSSL.
//! The connection is kept alive and transparently re-established.

use anyhow::{anyhow, bail, Context, Result};
use openssl::ssl::{SslConnector, SslMethod, SslOptions, SslStream, SslVerifyMode};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

pub struct Response {
    pub status: u16,
    #[allow(dead_code)]
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn json(&self) -> Result<serde_json::Value> {
        serde_json::from_slice(&self.body).with_context(|| {
            format!(
                "camera returned non-JSON response (HTTP {}): {}",
                self.status,
                String::from_utf8_lossy(&self.body[..self.body.len().min(200)])
            )
        })
    }
}

pub struct HttpsClient {
    host: String,
    port: u16,
    timeout: Duration,
    connector: SslConnector,
    conn: Option<SslStream<TcpStream>>,
}

impl HttpsClient {
    pub fn new(host: &str, port: u16, timeout: Duration) -> Result<Self> {
        let mut b = SslConnector::builder(SslMethod::tls_client())?;
        b.set_verify(SslVerifyMode::NONE);
        b.set_cipher_list("ALL:@SECLEVEL=0")?;
        // SSL_OP_LEGACY_SERVER_CONNECT (0x4) is not exported by the openssl crate.
        b.set_options(SslOptions::from_bits_retain(0x4) | SslOptions::ALLOW_UNSAFE_LEGACY_RENEGOTIATION);
        b.set_min_proto_version(None)?;
        Ok(Self { host: host.to_string(), port, timeout, connector: b.build(), conn: None })
    }

    pub fn authority(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    fn connect(&self) -> Result<SslStream<TcpStream>> {
        let addr = (self.host.as_str(), self.port).to_socket_addrs()?.next().ok_or_else(|| anyhow!("cannot resolve {}", self.host))?;
        let tcp = TcpStream::connect_timeout(&addr, self.timeout).with_context(|| format!("cannot connect to {}", self.authority()))?;
        tcp.set_read_timeout(Some(self.timeout))?;
        tcp.set_write_timeout(Some(self.timeout))?;
        tcp.set_nodelay(true)?;
        let cfg = self.connector.configure()?.verify_hostname(false).use_server_name_indication(false);
        cfg.connect(&self.host, tcp).map_err(|e| anyhow!("TLS handshake with {} failed: {e}", self.authority()))
    }

    /// POST `body` to `path`. Retries once on a fresh connection when a
    /// kept-alive connection turns out to be dead.
    pub fn post(&mut self, path: &str, headers: &[(&str, String)], body: &[u8]) -> Result<Response> {
        let reused = self.conn.is_some();
        match self.try_post(path, headers, body) {
            Ok(r) => Ok(r),
            Err(e) if reused => {
                log::debug!("request on kept-alive connection failed ({e:#}), reconnecting");
                self.conn = None;
                self.try_post(path, headers, body)
            }
            Err(e) => {
                self.conn = None;
                Err(e)
            }
        }
    }

    fn try_post(&mut self, path: &str, headers: &[(&str, String)], body: &[u8]) -> Result<Response> {
        if self.conn.is_none() {
            self.conn = Some(self.connect()?);
        }
        let authority = self.authority();
        let stream = self.conn.as_mut().unwrap();
        let mut req = format!("POST {path} HTTP/1.1\r\n");
        let mut has_host = false;
        for (k, v) in headers {
            if k.eq_ignore_ascii_case("host") {
                has_host = true;
            }
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        if !has_host {
            req.push_str(&format!("Host: {authority}\r\n"));
        }
        req.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        let mut out = req.into_bytes();
        out.extend_from_slice(body);
        stream.write_all(&out)?;
        stream.flush()?;
        let (resp, keep) = read_response(stream)?;
        if !keep {
            self.conn = None;
        }
        Ok(resp)
    }

    pub fn close(&mut self) {
        self.conn = None;
    }
}

fn read_response<S: Read>(s: &mut S) -> Result<(Response, bool)> {
    let mut buf = Vec::with_capacity(4096);
    let mut tmp = [0u8; 8192];
    let header_end = loop {
        if let Some(p) = find(&buf, b"\r\n\r\n") {
            break p + 4;
        }
        let n = s.read(&mut tmp)?;
        if n == 0 {
            bail!("connection closed before HTTP response headers");
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.len() > 64 * 1024 {
            bail!("HTTP response headers too large");
        }
    };
    let mut hdrs = [httparse::EMPTY_HEADER; 64];
    let mut parsed = httparse::Response::new(&mut hdrs);
    parsed.parse(&buf[..header_end]).map_err(|e| anyhow!("bad HTTP response: {e}"))?;
    let status = parsed.code.unwrap_or(0);
    let version = parsed.version.unwrap_or(1);
    let headers: Vec<(String, String)> =
        parsed.headers.iter().map(|h| (h.name.to_string(), String::from_utf8_lossy(h.value).trim().to_string())).collect();
    let get = |name: &str| headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.clone());
    let mut keep = version == 1;
    if let Some(c) = get("connection") {
        if c.eq_ignore_ascii_case("close") {
            keep = false;
        } else if c.eq_ignore_ascii_case("keep-alive") {
            keep = true;
        }
    }
    let mut rest = buf[header_end..].to_vec();
    let body = if get("transfer-encoding").map(|v| v.to_ascii_lowercase().contains("chunked")).unwrap_or(false) {
        read_chunked(s, rest)?
    } else if let Some(len) = get("content-length").and_then(|v| v.parse::<usize>().ok()) {
        while rest.len() < len {
            let n = s.read(&mut tmp)?;
            if n == 0 {
                bail!("connection closed in the middle of the HTTP body");
            }
            rest.extend_from_slice(&tmp[..n]);
        }
        rest.truncate(len);
        rest
    } else {
        keep = false;
        loop {
            match s.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => rest.extend_from_slice(&tmp[..n]),
                Err(e) if !rest.is_empty() && is_eof_like(&e) => break,
                Err(e) => return Err(e.into()),
            }
        }
        rest
    };
    Ok((Response { status, headers, body }, keep))
}

fn is_eof_like(e: &std::io::Error) -> bool {
    matches!(e.kind(), std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::ConnectionReset)
}

fn read_chunked<S: Read>(s: &mut S, mut buf: Vec<u8>) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut tmp = [0u8; 8192];
    let mut fill = |buf: &mut Vec<u8>| -> Result<()> {
        let n = s.read(&mut tmp)?;
        if n == 0 {
            bail!("connection closed inside chunked body");
        }
        buf.extend_from_slice(&tmp[..n]);
        Ok(())
    };
    loop {
        let line_end = loop {
            if let Some(p) = find(&buf, b"\r\n") {
                break p;
            }
            fill(&mut buf)?;
        };
        let line = String::from_utf8_lossy(&buf[..line_end]).to_string();
        let size =
            usize::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16).map_err(|_| anyhow!("bad chunk size {line:?}"))?;
        buf.drain(..line_end + 2);
        if size == 0 {
            // optional trailers until empty line
            loop {
                if let Some(p) = find(&buf, b"\r\n") {
                    if p == 0 {
                        return Ok(out);
                    }
                    buf.drain(..p + 2);
                } else {
                    // tolerate servers that close right after the last chunk
                    if fill(&mut buf).is_err() {
                        return Ok(out);
                    }
                }
            }
        }
        while buf.len() < size + 2 {
            fill(&mut buf)?;
        }
        out.extend_from_slice(&buf[..size]);
        buf.drain(..size + 2);
    }
}

pub fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_length_and_chunked() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhelloEXTRA";
        let (r, keep) = read_response(&mut &raw[..]).unwrap();
        assert_eq!(r.body, b"hello");
        assert!(keep);
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n";
        let (r, keep) = read_response(&mut &raw[..]).unwrap();
        assert_eq!(r.body, b"abcde");
        assert!(!keep);
        let raw = b"HTTP/1.0 401 Unauthorized\r\n\r\n{\"a\":1}";
        let (r, keep) = read_response(&mut &raw[..]).unwrap();
        assert_eq!(r.status, 401);
        assert_eq!(r.body, b"{\"a\":1}");
        assert!(!keep);
    }
}
