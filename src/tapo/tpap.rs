//! TPAP transport: the SPAKE2+ login and the AES-128-CCM `/stok=<stok>/ds`
//! channel used by recent Tapo camera firmware (port of pytapo's
//! `transport/tpap/tpap.py`).

use super::crypto::{md5_hex, sha256_hex_upper};
use super::https::HttpsClient;
use super::spake2p::{apply_extra_crypt, RegisterResult, Spake2pClient};
use super::{pyjson, TapoError, Transport};
use aes::Aes128;
use anyhow::{anyhow, bail, Result};
use base64::Engine as _;
use ccm::aead::{Aead, KeyInit};
use ccm::consts::{U12, U16};
use ccm::Ccm;
use num_bigint::BigUint;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

type Aes128Ccm = Ccm<Aes128, U16, U12>;
const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;
const CAMERA_USER: &str = "admin";
const SESSION_ERRORS: &[i64] = &[-40401, -40421];

pub struct Tpap {
    http: HttpsClient,
    password: String,
    info: Option<Value>,
    stok: Option<String>,
    seq: u32,
    key: [u8; 16],
    nonce: [u8; 12],
    expires: Option<Instant>,
    passcode_index: Option<usize>,
}

fn login_headers(http: &HttpsClient, content_type: &str) -> Vec<(&'static str, String)> {
    vec![
        ("requestByApp", "true".into()),
        ("Referer", format!("https://{}", http.authority())),
        ("User-Agent", "Tapo CameraClient Android".into()),
        ("Content-Type", content_type.into()),
        ("Accept", content_type.into()),
    ]
}

fn post_login(http: &mut HttpsClient, params: Value) -> Result<Value> {
    let headers = login_headers(http, "application/json");
    let body = pyjson::dumps(&json!({"method": "login", "params": params}));
    http.post("/", &headers, body.as_bytes())?.json()
}

/// The camera's TPAP info if it offers the SPAKE2+ password login.
pub fn discover(http: &mut HttpsClient) -> Option<Value> {
    let r = post_login(http, json!({"sub_method": "discover"})).ok()?;
    let tpap = r.pointer("/result/tpap")?.clone();
    let pake = tpap.get("pake")?.as_array()?;
    pake.iter().any(|v| v.as_i64() == Some(2)).then_some(tpap)
}

/// True if the camera answers the legacy encrypt_type 3 probe with -40211
/// (it then only speaks TPAP). The probe carries no password.
pub fn rejects_legacy_login(http: &mut HttpsClient) -> bool {
    let cnonce = super::crypto::random_hex(8).to_uppercase();
    post_login(http, json!({"encrypt_type": "3", "username": CAMERA_USER, "cnonce": cnonce}))
        .map(|r| r.get("error_code").and_then(Value::as_i64) == Some(-40211))
        .unwrap_or(false)
}

fn lockout_seconds(r: &Value) -> i64 {
    for p in ["/error_info/sec_left", "/data/sec_left", "/result/data/sec_left"] {
        if let Some(v) = r.pointer(p) {
            let n = v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).unwrap_or(0);
            if n > 0 {
                return n;
            }
        }
    }
    0
}

#[derive(Debug)]
struct TpapError {
    code: Option<i64>,
    msg: String,
}

impl std::fmt::Display for TpapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TPAP error_code={:?} {}", self.code, self.msg)
    }
}
impl std::error::Error for TpapError {}

fn tpap_err(code: Option<i64>, msg: &str) -> anyhow::Error {
    anyhow::Error::new(TpapError { code, msg: msg.to_string() })
}

fn tpap_code(e: &anyhow::Error) -> Option<Option<i64>> {
    e.downcast_ref::<TpapError>().map(|t| t.code)
}

impl Tpap {
    pub fn new(http: HttpsClient, cloud_password: &str, info: Option<Value>) -> Self {
        Self {
            http,
            password: cloud_password.to_string(),
            info,
            stok: None,
            seq: 0,
            key: [0; 16],
            nonce: [0; 12],
            expires: None,
            passcode_index: None,
        }
    }

    fn logged_in(&self) -> bool {
        self.stok.is_some() && self.expires.map(|e| Instant::now() < e).unwrap_or(false)
    }

    fn drop_session(&mut self) {
        self.stok = None;
        self.expires = None;
    }

    fn username(&self) -> String {
        if self.info.as_ref().and_then(|i| i.get("user_hash_type")).and_then(Value::as_i64) == Some(1) {
            sha256_hex_upper(CAMERA_USER)
        } else {
            md5_hex(CAMERA_USER)
        }
    }

    fn passcodes(&self) -> [String; 2] {
        [md5_hex(&self.password), sha256_hex_upper(&self.password)]
    }

    fn login(&mut self) -> Result<()> {
        if self.password.is_empty() {
            bail!(TapoError::InvalidAuth);
        }
        if self.info.is_none() {
            self.info = Some(discover(&mut self.http).unwrap_or_else(|| json!({})));
        }
        let passcodes = self.passcodes();
        // A re-login only uses the passcode that worked before: every failed
        // attempt counts towards the camera's lockout.
        let attempts: Vec<usize> = match self.passcode_index {
            Some(i) => vec![i, i],
            None => (0..passcodes.len()).collect(),
        };
        let mut last: Option<anyhow::Error> = None;
        for (n, &i) in attempts.iter().enumerate() {
            if n > 0 && attempts[n - 1] == i {
                std::thread::sleep(Duration::from_secs(2));
            }
            match self.login_with(&passcodes[i]) {
                Ok(()) => {
                    self.passcode_index = Some(i);
                    return Ok(());
                }
                Err(e) => {
                    if tpap_code(&e) != Some(Some(-40401)) {
                        return Err(e);
                    }
                    last = Some(e);
                }
            }
        }
        log::debug!("TPAP login failed: {:?}", last);
        bail!(TapoError::InvalidAuth)
    }

    fn login_with(&mut self, passcode: &str) -> Result<()> {
        let mut user_random = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut user_random);
        let user_random_b64 = B64.encode(user_random);
        let username = self.username();
        let register = post_login(
            &mut self.http,
            json!({
                "sub_method": "pake_register",
                "username": username,
                "user_random": user_random_b64,
                "cipher_suites": [1],
                "encryption": ["aes_128_ccm"],
                "passcode_type": "userpw",
            }),
        )?;
        let Some(result) = register.get("result") else {
            return Err(tpap_err(register.get("error_code").and_then(Value::as_i64), "pake_register failed"));
        };
        let credential = apply_extra_crypt(passcode, result.get("extra_crypt"))?;
        let field = |k: &str| -> Result<Vec<u8>> {
            let s = result.get(k).and_then(Value::as_str).ok_or_else(|| anyhow!("pake_register reply misses {k}"))?;
            Ok(B64.decode(s)?)
        };
        let iterations = result
            .get("iterations")
            .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
            .ok_or_else(|| anyhow!("pake_register reply misses iterations"))? as u32;
        let reg =
            RegisterResult { dev_salt: field("dev_salt")?, iterations, dev_share: field("dev_share")?, dev_random: field("dev_random")? };
        let mut rnd = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut rnd);
        let client = Spake2pClient::new(&reg, &user_random, &credential, &BigUint::from_bytes_be(&rnd))?;
        let share = post_login(
            &mut self.http,
            json!({
                "sub_method": "pake_share",
                "user_share": B64.encode(&client.xb),
                "user_confirm": B64.encode(&client.user_confirm),
            }),
        )?;
        let Some(sr) = share.get("result") else {
            let left = lockout_seconds(&share);
            if left > 0 {
                bail!(TapoError::Suspended(left));
            }
            return Err(tpap_err(share.get("error_code").and_then(Value::as_i64), "pake_share failed"));
        };
        let dev_confirm = B64.decode(sr.get("dev_confirm").and_then(Value::as_str).unwrap_or(""))?;
        if !client.device_confirm_ok(&dev_confirm) {
            return Err(tpap_err(None, "device confirmation mismatch"));
        }
        let (key, nonce) = client.session_key_and_nonce();
        self.key = key;
        self.nonce = nonce;
        self.stok = Some(sr.get("stok").and_then(Value::as_str).ok_or_else(|| anyhow!("no stok"))?.to_string());
        self.seq = sr.get("start_seq").and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).unwrap_or(0) as u32;
        let ttl = sr.get("expired").and_then(Value::as_u64).filter(|&t| t > 0).unwrap_or(3600);
        self.expires = Some(Instant::now() + Duration::from_secs(ttl));
        Ok(())
    }

    fn nonce_for(&self, seq: u32) -> [u8; 12] {
        let mut n = [0u8; 12];
        n[..8].copy_from_slice(&self.nonce[..8]);
        n[8..].copy_from_slice(&seq.to_be_bytes());
        n
    }

    fn round_trip(&mut self, payload: &[u8]) -> Result<Value> {
        let seq = self.seq;
        self.seq = self.seq.wrapping_add(1);
        let cipher = Aes128Ccm::new(&self.key.into());
        let ct = cipher.encrypt(&self.nonce_for(seq).into(), payload).map_err(|_| anyhow!("AES-CCM encryption failed"))?;
        let mut body = seq.to_be_bytes().to_vec();
        body.extend_from_slice(&ct);
        let headers = login_headers(&self.http, "application/octet-stream");
        let path = format!("/stok={}/ds", self.stok.as_deref().unwrap_or_default());
        let res = self.http.post(&path, &headers, &body)?;
        let raw = res.body;
        if raw.first() == Some(&b'{') {
            let code = serde_json::from_slice::<Value>(&raw).ok().and_then(|v| v.get("error_code").and_then(Value::as_i64));
            return Err(tpap_err(code, "request refused"));
        }
        if raw.len() < 20 {
            return Err(tpap_err(None, &format!("short reply ({} bytes, HTTP {})", raw.len(), res.status)));
        }
        let reply_seq = u32::from_be_bytes(raw[..4].try_into().unwrap());
        let plain =
            cipher.decrypt(&self.nonce_for(reply_seq).into(), &raw[4..]).map_err(|_| tpap_err(None, "reply failed authentication"))?;
        Ok(serde_json::from_slice(&plain)?)
    }
}

impl Transport for Tpap {
    fn send(&mut self, request: &Value) -> Result<Value> {
        // The /ds channel only takes multipleRequest; wrap single calls.
        let single = request.get("method").and_then(Value::as_str) != Some("multipleRequest");
        let inner = if single { json!({"method": "multipleRequest", "params": {"requests": [request]}}) } else { request.clone() };
        let payload = pyjson::dumps_compact(&inner);
        let mut response = None;
        for attempt in 0..2 {
            let r = (|| {
                if !self.logged_in() {
                    self.login()?;
                }
                self.round_trip(payload.as_bytes())
            })();
            match r {
                Ok(v) => {
                    response = Some(v);
                    break;
                }
                Err(e) => {
                    let retry = match tpap_code(&e) {
                        Some(Some(code)) => SESSION_ERRORS.contains(&code),
                        Some(None) => false,
                        None => e.downcast_ref::<TapoError>().is_none(), // network error
                    };
                    self.drop_session();
                    if attempt == 1 || !retry {
                        return Err(e);
                    }
                }
            }
        }
        let response = response.unwrap();
        if single {
            if let Some(first) = response.pointer("/result/responses/0") {
                return Ok(first.clone());
            }
        }
        Ok(response)
    }

    fn name(&self) -> &'static str {
        "tpap (SPAKE2+)"
    }
}
