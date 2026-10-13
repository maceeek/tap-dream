//! SPAKE2+ (P-256, RFC 9383 M/N points) and the credential transforms used by
//! the TP-Link "TPAP" login of newer Tapo firmware. Port of pytapo's
//! `transport/tpap/spake2p.py` (itself based on freeKC/tapo-v4-protocol, MIT).

use anyhow::{anyhow, bail, Result};
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use num_bigint::BigUint;
use p256::elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint};
use p256::elliptic_curve::PrimeField;
use p256::{AffinePoint, EncodedPoint, FieldBytes, ProjectivePoint, Scalar};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

const N_HEX: &str = "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551";
const M_SEC1: &str = "02886e2f97ace46e55ba9dd7242579f2993b64e16ef3dcab95afd497333d8fa12f";
const N_SEC1: &str = "03d8bbd6c639c62937b04d997f38c3770719c629d7014d49a24b4f98baa1292b49";
const CONTEXT_TAG: &[u8] = b"PAKE V1";

fn order() -> BigUint {
    BigUint::parse_bytes(N_HEX.as_bytes(), 16).unwrap()
}

pub fn decode_point(b: &[u8]) -> Result<ProjectivePoint> {
    let ep = EncodedPoint::from_bytes(b).map_err(|e| anyhow!("bad point encoding: {e}"))?;
    let p: Option<AffinePoint> = AffinePoint::from_encoded_point(&ep).into();
    Ok(ProjectivePoint::from(p.ok_or_else(|| anyhow!("point not on curve"))?))
}

pub fn encode_point(p: &ProjectivePoint) -> Vec<u8> {
    p.to_affine().to_encoded_point(false).as_bytes().to_vec()
}

fn scalar_from_biguint(v: &BigUint) -> Scalar {
    let v = v % order();
    let mut bytes = v.to_bytes_be();
    let mut repr = vec![0u8; 32 - bytes.len()];
    repr.append(&mut bytes);
    Option::from(Scalar::from_repr(*FieldBytes::from_slice(&repr))).expect("reduced scalar")
}

fn scalar_to_bytes(s: &Scalar) -> Vec<u8> {
    s.to_repr().to_vec()
}

fn hkdf_sha256(ikm: &[u8], salt: Option<&[u8]>, info: &[u8], len: usize) -> Vec<u8> {
    let hk = Hkdf::<Sha256>::new(salt, ikm);
    let mut out = vec![0u8; len];
    hk.expand(info, &mut out).expect("hkdf length");
    out
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut m = <HmacSha256 as Mac>::new_from_slice(key).unwrap();
    m.update(data);
    m.finalize().into_bytes().to_vec()
}

fn len_prefixed(chunks: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for c in chunks {
        out.extend_from_slice(&(c.len() as u64).to_le_bytes());
        out.extend_from_slice(c);
    }
    out
}

// ------------------------------------------------------------------ credentials

const CRYPT_B64: &[u8] = b"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const CRYPT_ORDER: [(usize, usize, usize); 10] =
    [(0, 10, 20), (21, 1, 11), (12, 22, 2), (3, 13, 23), (24, 4, 14), (15, 25, 5), (6, 16, 26), (27, 7, 17), (18, 28, 8), (9, 19, 29)];

fn to64(mut v: u32, n: usize, out: &mut String) {
    for _ in 0..n {
        out.push(CRYPT_B64[(v & 0x3f) as usize] as char);
        v >>= 6;
    }
}

fn repeat(digest: &[u8], len: usize) -> Vec<u8> {
    digest.iter().cycle().take(len).copied().collect()
}

/// SHA-256-crypt ("$5$", Drepper) exactly as the Tapo app's `sha256Shadow`.
pub fn sha256_crypt(key: &str, prefix: &str) -> String {
    let mut rest = prefix.strip_prefix("$5$").unwrap_or(prefix);
    let (mut rounds, mut explicit) = (5000u64, false);
    if let Some(r) = rest.strip_prefix("rounds=") {
        if let Some((head, tail)) = r.split_once('$').filter(|(_, t)| !t.is_empty()) {
            if let Ok(n) = head.parse::<u64>() {
                rounds = n.clamp(1000, 999_999_999);
                rest = tail;
                explicit = true;
            }
        }
    }
    let end = rest.find('$').filter(|&e| e > 0).unwrap_or(rest.len());
    let salt: String = rest[..end].chars().take(16).collect();
    let (k, s) = (key.as_bytes(), salt.as_bytes());

    let b = Sha256::digest([k, s, k].concat());
    let mut a = Sha256::new();
    a.update(k);
    a.update(s);
    a.update(repeat(&b, k.len()));
    let mut n = k.len();
    while n > 0 {
        if n & 1 == 1 {
            a.update(b);
        } else {
            a.update(k);
        }
        n >>= 1;
    }
    let a = a.finalize();
    let p = repeat(&Sha256::digest(k.repeat(k.len())), k.len());
    let sb = repeat(&Sha256::digest(s.repeat(16 + a[0] as usize)), s.len());

    let mut c = a.to_vec();
    for i in 0..rounds {
        let mut h = Sha256::new();
        if i & 1 == 1 {
            h.update(&p);
        } else {
            h.update(&c);
        }
        if i % 3 != 0 {
            h.update(&sb);
        }
        if i % 7 != 0 {
            h.update(&p);
        }
        if i & 1 == 1 {
            h.update(&c);
        } else {
            h.update(&p);
        }
        c = h.finalize().to_vec();
    }
    let mut enc = String::new();
    for (x, y, z) in CRYPT_ORDER {
        to64(((c[x] as u32) << 16) | ((c[y] as u32) << 8) | c[z] as u32, 4, &mut enc);
    }
    to64(((c[31] as u32) << 8) | c[30] as u32, 3, &mut enc);
    let rounds_part = if explicit { format!("rounds={rounds}$") } else { String::new() };
    format!("$5${rounds_part}{salt}${enc}")
}

/// Turn a passcode into the credential the camera asked for in `pake_register`.
pub fn apply_extra_crypt(passcode: &str, extra: Option<&serde_json::Value>) -> Result<String> {
    let Some(extra) = extra.filter(|v| !v.is_null() && v.as_object().map(|o| !o.is_empty()).unwrap_or(true)) else {
        return Ok(passcode.to_string());
    };
    let kind = extra.get("type").and_then(|v| v.as_str()).unwrap_or("").to_lowercase();
    let params = extra.get("params").cloned().unwrap_or_default();
    if kind == "password_shadow" {
        let id = match params.get("passwd_id") {
            Some(serde_json::Value::Number(n)) => n.as_i64().unwrap_or(0),
            Some(serde_json::Value::String(s)) => s.parse().unwrap_or(0),
            _ => 0,
        };
        let prefix = match params.get("passwd_prefix") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(v) if !v.is_null() => v.to_string(),
            _ => String::new(),
        };
        return match id {
            5 => Ok(sha256_crypt(passcode, &prefix)),
            2 => Ok(hex::encode(sha1::Sha1::digest(passcode.as_bytes()))),
            other => bail!("unsupported password_shadow passwd_id {other}"),
        };
    }
    bail!("unsupported extra_crypt type {kind:?}")
}

// ------------------------------------------------------------------ the exchange

pub struct RegisterResult {
    pub dev_salt: Vec<u8>,
    pub iterations: u32,
    pub dev_share: Vec<u8>,
    pub dev_random: Vec<u8>,
}

pub struct Spake2pClient {
    pub xb: Vec<u8>,
    pub user_confirm: Vec<u8>,
    kcb: Vec<u8>,
    shared_key: Vec<u8>,
}

impl Spake2pClient {
    /// `random` is reduced like pytapo does: x = random % (N-1) + 1.
    pub fn new(reg: &RegisterResult, user_random: &[u8], credential: &str, random: &BigUint) -> Result<Self> {
        let mut dk = [0u8; 80];
        pbkdf2::pbkdf2_hmac::<Sha256>(credential.as_bytes(), &reg.dev_salt, reg.iterations, &mut dk);
        let w0 = scalar_from_biguint(&BigUint::from_bytes_be(&dk[..40]));
        let w1 = scalar_from_biguint(&BigUint::from_bytes_be(&dk[40..]));
        let m = decode_point(&hex::decode(M_SEC1).unwrap())?;
        let n = decode_point(&hex::decode(N_SEC1).unwrap())?;
        let y = decode_point(&reg.dev_share)?;
        let x = scalar_from_biguint(&(random % (order() - 1u32) + 1u32));
        let big_x = ProjectivePoint::GENERATOR * x + m * w0;
        let h = y - n * w0;
        let xb = encode_point(&big_x);
        let yb = encode_point(&y);
        let context = Sha256::digest([CONTEXT_TAG, user_random, &reg.dev_random].concat());
        let transcript = len_prefixed(&[
            &context,
            b"",
            b"",
            &encode_point(&m),
            &encode_point(&n),
            &xb,
            &yb,
            &encode_point(&(h * x)),
            &encode_point(&(h * w1)),
            &scalar_to_bytes(&w0),
        ]);
        let ke = Sha256::digest(&transcript);
        let conf = hkdf_sha256(&ke, None, b"ConfirmationKeys", 64);
        let user_confirm = hmac(&conf[..32], &yb);
        let shared_key = hkdf_sha256(&ke, None, b"SharedKey", 32);
        Ok(Self { xb, user_confirm, kcb: conf[32..].to_vec(), shared_key })
    }

    pub fn device_confirm_ok(&self, dev_confirm: &[u8]) -> bool {
        let mut m = <HmacSha256 as Mac>::new_from_slice(&self.kcb).unwrap();
        m.update(&self.xb);
        m.verify_slice(dev_confirm).is_ok()
    }

    pub fn session_key_and_nonce(&self) -> ([u8; 16], [u8; 12]) {
        let key = hkdf_sha256(&self.shared_key, Some(b"tp-kdf-salt-aes128-key"), b"tp-kdf-info-aes128-key", 32);
        let nonce = hkdf_sha256(&self.shared_key, Some(b"tp-kdf-salt-aes128-iv"), b"tp-kdf-info-aes128-iv", 32);
        (key[..16].try_into().unwrap(), nonce[..12].try_into().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    #[derive(serde::Deserialize)]
    struct Vectors {
        crypt: Vec<(String, String, String)>,
        sha1: (String, String),
        dev_salt: String,
        iterations: u32,
        dev_share: String,
        dev_random: String,
        user_random: String,
        credential: String,
        random: String,
        xb: String,
        user_confirm: String,
        dev_confirm: String,
        key: String,
        nonce: String,
    }

    /// Vectors generated with pytapo's spake2p.py (see tools/gen_test_vectors.py).
    #[test]
    fn matches_pytapo() {
        let v: Vectors = serde_json::from_str(include_str!("testdata/spake2p.json")).unwrap();
        for (key, prefix, expected) in &v.crypt {
            assert_eq!(&sha256_crypt(key, prefix), expected, "crypt({key}, {prefix})");
        }
        let extra = serde_json::json!({"type": "password_shadow", "params": {"passwd_id": 2}});
        assert_eq!(apply_extra_crypt(&v.sha1.0, Some(&extra)).unwrap(), v.sha1.1);
        let b64 = base64::engine::general_purpose::STANDARD;
        let reg = RegisterResult {
            dev_salt: b64.decode(&v.dev_salt).unwrap(),
            iterations: v.iterations,
            dev_share: b64.decode(&v.dev_share).unwrap(),
            dev_random: b64.decode(&v.dev_random).unwrap(),
        };
        let random = BigUint::parse_bytes(v.random.as_bytes(), 10).unwrap();
        let c = Spake2pClient::new(&reg, &b64.decode(&v.user_random).unwrap(), &v.credential, &random).unwrap();
        assert_eq!(b64.encode(&c.xb), v.xb);
        assert_eq!(b64.encode(&c.user_confirm), v.user_confirm);
        assert!(c.device_confirm_ok(&b64.decode(&v.dev_confirm).unwrap()));
        let (key, nonce) = c.session_key_and_nonce();
        assert_eq!(hex::encode(key), v.key);
        assert_eq!(hex::encode(nonce), v.nonce);
    }
}
