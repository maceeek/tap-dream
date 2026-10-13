//! The "classic" control transport (port of pytapo's `transport/pytapo`):
//! either the old hashed-password login, or the encrypt_type 3 login with
//! AES-CBC `securePassthrough` requests and the `Tapo_tag` signature.

use super::crypto::{aes_cbc_decrypt, aes_cbc_encrypt, md5_hex_upper, random_hex, sha256, sha256_hex_upper, sha256_hex_upper_bytes};
use super::https::HttpsClient;
use super::{pyjson, TapoError, Transport};
use anyhow::{anyhow, bail, Result};
use base64::Engine as _;
use serde_json::{json, Value};

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

/// Error codes after which pytapo drops the session and retries once.
const RETRYABLE: &[i64] = &[1002, 1112, -1, -1001, 9999, -40401, -40413, -40421, -40210];
const AUTH_ERRORS: &[i64] = &[-1501, 1111, -1005, 1100, 1003, -40412];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HashMethod {
    Md5,
    Sha256,
}

pub struct Classic {
    http: HttpsClient,
    user: String,
    hashed_md5: String,
    hashed_sha256: String,
    secure: Option<bool>,
    method: Option<HashMethod>,
    cnonce: String,
    lsk: [u8; 16],
    ivb: [u8; 16],
    seq: Option<i64>,
    stok: Option<String>,
}

impl Classic {
    pub fn new(http: HttpsClient, user: &str, password: &str) -> Self {
        Self {
            http,
            user: user.to_string(),
            hashed_md5: md5_hex_upper(password),
            hashed_sha256: sha256_hex_upper(password),
            secure: None,
            method: None,
            cnonce: String::new(),
            lsk: [0; 16],
            ivb: [0; 16],
            seq: None,
            stok: None,
        }
    }

    fn headers(&self) -> Vec<(&'static str, String)> {
        let authority = self.http.authority();
        vec![
            ("Host", authority.clone()),
            ("Referer", format!("https://{authority}")),
            ("Accept", "application/json".into()),
            ("User-Agent", "Tapo CameraClient Android".into()),
            ("Connection", "keep-alive".into()),
            ("requestByApp", "true".into()),
            ("Content-Type", "application/json; charset=UTF-8".into()),
        ]
    }

    fn post_root(&mut self, data: &Value) -> Result<(u16, Value)> {
        let headers = self.headers();
        let res = self.http.post("/", &headers, pyjson::dumps(data).as_bytes())?;
        let status = res.status;
        Ok((status, res.json()?))
    }

    fn hashed_password(&self) -> Result<&str> {
        match self.method {
            Some(HashMethod::Md5) => Ok(&self.hashed_md5),
            Some(HashMethod::Sha256) => Ok(&self.hashed_sha256),
            None => bail!("Failure detecting hashing algorithm."),
        }
    }

    fn clear_session(&mut self) {
        self.stok = None;
        self.seq = None;
        self.method = None;
        self.secure = None;
        self.http.close();
    }

    pub fn is_secure(&mut self) -> Result<bool> {
        if let Some(s) = self.secure {
            return Ok(s);
        }
        let probe = random_hex(8).to_uppercase();
        let (_, r) = self.post_root(&json!({
            "method": "login",
            "params": {"encrypt_type": "3", "username": self.user, "cnonce": probe}
        }))?;
        let secure = r.get("error_code").and_then(Value::as_i64) == Some(-40413)
            && match r.pointer("/result/data/encrypt_type") {
                Some(Value::String(s)) => s.contains('3'),
                Some(Value::Array(a)) => a.iter().any(|v| v.as_str() == Some("3")),
                _ => false,
            };
        self.secure = Some(secure);
        Ok(secure)
    }

    /// Port of `_validateDeviceConfirm`.
    pub fn validate_device_confirm(&mut self, nonce: &str, device_confirm: &str) -> bool {
        let with =
            |hashed: &str| sha256_hex_upper_bytes(&[self.cnonce.as_bytes(), hashed.as_bytes(), nonce.as_bytes()]) + nonce + &self.cnonce;
        self.method = if device_confirm == with(&self.hashed_sha256) {
            Some(HashMethod::Sha256)
        } else if device_confirm == with(&self.hashed_md5) {
            Some(HashMethod::Md5)
        } else {
            None
        };
        self.method.is_some()
    }

    /// Port of `_generateEncryptionToken` ("lsk" / "ivb").
    pub fn encryption_token(&self, kind: &str, nonce: &str) -> Result<[u8; 16]> {
        let hashed_key = sha256_hex_upper_bytes(&[self.cnonce.as_bytes(), self.hashed_password()?.as_bytes(), nonce.as_bytes()]);
        let d = sha256(format!("{kind}{}{nonce}{hashed_key}", self.cnonce).as_bytes());
        Ok(d[..16].try_into().unwrap())
    }

    /// Port of `_getTag`.
    pub fn tag(&self, body: &str, seq: i64) -> Result<String> {
        let t = sha256_hex_upper_bytes(&[self.hashed_password()?.as_bytes(), self.cnonce.as_bytes()]);
        Ok(sha256_hex_upper_bytes(&[t.as_bytes(), body.as_bytes(), seq.to_string().as_bytes()]))
    }

    fn check_suspension(r: &Value) -> Result<()> {
        for p in ["/result/data", "/data"] {
            if let Some(left) = r.pointer(&format!("{p}/sec_left")).and_then(Value::as_i64) {
                if left > 0 {
                    bail!(TapoError::Suspended(left));
                }
            }
        }
        Ok(())
    }

    fn refresh_stok(&mut self, attempt: u32) -> Result<()> {
        self.cnonce = random_hex(8).to_uppercase();
        let secure = self.is_secure()?;
        let data = if secure {
            json!({"method": "login", "params": {"cnonce": self.cnonce, "encrypt_type": "3", "username": self.user}})
        } else {
            json!({"method": "login", "params": {"hashed": true, "password": self.hashed_md5, "username": self.user}})
        };
        let (status, mut r) = self.post_root(&data)?;
        if status == 401 && r.pointer("/result/data/code").and_then(Value::as_i64) == Some(-40411) {
            bail!(TapoError::InvalidAuth);
        }
        if secure {
            let nonce = r.pointer("/result/data/nonce").and_then(Value::as_str).map(str::to_string);
            let confirm = r.pointer("/result/data/device_confirm").and_then(Value::as_str).map(str::to_string);
            match (nonce, confirm) {
                (Some(nonce), Some(confirm)) => {
                    if !self.validate_device_confirm(&nonce, &confirm) {
                        if attempt < 1 && r.get("error_code").and_then(Value::as_i64) == Some(-40413) {
                            return self.refresh_stok(attempt + 1);
                        }
                        bail!(TapoError::InvalidAuth);
                    }
                    let digest = sha256_hex_upper_bytes(&[self.hashed_password()?.as_bytes(), self.cnonce.as_bytes(), nonce.as_bytes()]);
                    let data = json!({"method": "login", "params": {
                        "cnonce": self.cnonce,
                        "encrypt_type": "3",
                        "digest_passwd": format!("{digest}{}{nonce}", self.cnonce),
                        "username": self.user,
                    }});
                    let (_, r2) = self.post_root(&data)?;
                    r = r2;
                    if let Some(start_seq) = r.pointer("/result/start_seq").and_then(Value::as_i64) {
                        if let Some(group) = r.pointer("/result/user_group").and_then(Value::as_str) {
                            if group != "root" {
                                // encrypted control via a 3rd party account is not supported by the camera
                                bail!(TapoError::InvalidAuth);
                            }
                        }
                        self.lsk = self.encryption_token("lsk", &nonce)?;
                        self.ivb = self.encryption_token("ivb", &nonce)?;
                        self.seq = Some(start_seq);
                    }
                }
                _ => {
                    Self::check_suspension(&r)?;
                    if attempt < 1 && r.get("error_code").and_then(Value::as_i64) == Some(-40413) {
                        return self.refresh_stok(attempt + 1);
                    }
                    bail!(TapoError::InvalidAuth);
                }
            }
        } else {
            self.method = Some(HashMethod::Md5);
        }
        Self::check_suspension(&r)?;
        let code = r.get("error_code").and_then(Value::as_i64).unwrap_or(0);
        if code == 0 {
            if let Some(stok) = r.pointer("/result/stok").and_then(Value::as_str) {
                self.stok = Some(stok.to_string());
                return Ok(());
            }
        }
        if attempt < 1 && code != 0 && !AUTH_ERRORS.contains(&code) && code != -40411 {
            log::debug!("login returned {code}, retrying");
            self.clear_session();
            std::thread::sleep(std::time::Duration::from_secs(1));
            return self.refresh_stok(attempt + 1);
        }
        log::debug!("login failed: {r}");
        bail!(TapoError::InvalidAuth)
    }

    fn send_once(&mut self, request: &Value) -> Result<SendOutcome> {
        if self.stok.is_none() {
            self.refresh_stok(0)?;
        }
        let path = format!("/stok={}/ds", self.stok.as_deref().unwrap_or_default());
        let secure = self.is_secure()?;
        let mut headers = self.headers();
        let body = if let (true, Some(seq)) = (secure, self.seq) {
            let inner = pyjson::dumps(request);
            let enc = aes_cbc_encrypt(&self.lsk, &self.ivb, inner.as_bytes());
            let body = pyjson::dumps(&json!({
                "method": "securePassthrough",
                "params": {"request": B64.encode(enc)}
            }));
            headers.push(("Seq", seq.to_string()));
            headers.push(("Tapo_tag", self.tag(&body, seq)?));
            self.seq = Some(seq + 1);
            body
        } else {
            pyjson::dumps(request)
        };
        let res = self.http.post(&path, &headers, body.as_bytes())?;
        let data = res.json()?;
        let response = if secure {
            match data.pointer("/result/response").and_then(Value::as_str) {
                Some(enc) => {
                    let raw = B64.decode(enc).map_err(|e| anyhow!("bad base64 from camera: {e}"))?;
                    match aes_cbc_decrypt(&self.lsk, &self.ivb, &raw) {
                        Ok(plain) => serde_json::from_slice(&plain)?,
                        Err(_) => return Ok(SendOutcome::Reauth(data)),
                    }
                }
                None => data,
            }
        } else {
            data
        };
        let code = response.get("error_code").and_then(Value::as_i64);
        if let Some(code) = code {
            if RETRYABLE.contains(&code) {
                return Ok(SendOutcome::Reauth(response));
            }
            if AUTH_ERRORS.contains(&code) {
                self.clear_session();
            }
        }
        Ok(SendOutcome::Done(response))
    }
}

enum SendOutcome {
    Done(Value),
    Reauth(Value),
}

impl Transport for Classic {
    fn send(&mut self, request: &Value) -> Result<Value> {
        let mut last = Value::Null;
        for attempt in 0..2 {
            match self.send_once(request) {
                Ok(SendOutcome::Done(v)) => return Ok(v),
                Ok(SendOutcome::Reauth(v)) => {
                    log::debug!("session needs re-authentication ({})", v.get("error_code").unwrap_or(&Value::Null));
                    last = v;
                }
                Err(e) if attempt == 0 && e.downcast_ref::<TapoError>().is_none() => {
                    log::debug!("request failed: {e:#}, retrying with a new session");
                }
                Err(e) => return Err(e),
            }
            self.clear_session();
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        Ok(last)
    }

    fn name(&self) -> &'static str {
        if self.secure == Some(true) {
            "classic (encrypt_type 3)"
        } else {
            "classic"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client() -> Classic {
        let http = HttpsClient::new("127.0.0.1", 443, std::time::Duration::from_secs(1)).unwrap();
        let mut c = Classic::new(http, "admin", "secret");
        c.cnonce = "ABCDEF0123456789".into();
        c
    }

    /// Vectors computed with pytapo's Python implementation.
    #[test]
    fn matches_pytapo_vectors() {
        let mut c = client();
        let nonce = "0011223344556677";
        let confirm = sha256_hex_upper_bytes(&[c.cnonce.as_bytes(), c.hashed_sha256.as_bytes(), nonce.as_bytes()]) + nonce + &c.cnonce;
        assert!(c.validate_device_confirm(nonce, &confirm));
        assert_eq!(c.method, Some(HashMethod::Sha256));
        assert_eq!(hex::encode(c.encryption_token("lsk", nonce).unwrap()), include_str!("testdata/lsk.txt").trim());
        assert_eq!(hex::encode(c.encryption_token("ivb", nonce).unwrap()), include_str!("testdata/ivb.txt").trim());
        assert_eq!(c.tag("{\"a\": 1}", 42).unwrap(), include_str!("testdata/tag.txt").trim());
    }
}
