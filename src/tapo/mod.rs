//! Minimal Tapo camera client – just what is needed to list and download SD
//! card recordings. Ported from pytapo (MIT, Juraj Nyíri and contributors).

pub mod classic;
pub mod crypto;
pub mod https;
pub mod media;
pub mod pyjson;
pub mod spake2p;
pub mod tpap;
pub mod ts;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

pub const CONNECTION_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, thiserror::Error)]
pub enum TapoError {
    #[error("invalid authentication data (check the cloud password; the camera locks out after repeated failures)")]
    InvalidAuth,
    #[error("camera temporarily blocks logins, try again in {0} s")]
    Suspended(i64),
    #[error("camera error {code}: {msg}")]
    Camera { code: i64, msg: String },
}

pub trait Transport: Send {
    fn send(&mut self, request: &Value) -> Result<Value>;
    fn name(&self) -> &'static str;
}

pub fn error_message(code: i64) -> &'static str {
    match code {
        -40401 => "Invalid stok value",
        -64324 => "Privacy mode is ON, not able to execute",
        -40105 => "Method does not exist",
        -40209 => "Invalid login credentials",
        -71103 => "User ID is not authorized (USER_ID_INVALID)",
        -71105 => "PLAYBACK_SEARCH_FAILED",
        -71101 => "USER_ID_FULL",
        -71102 => "USER_ID_EMPLOYED",
        -52409 => "SD_CARD_UNPLUGGED",
        -71114 => "STORAGE_NOT_EXIST",
        -71115 => "STORAGE_SD_LOADING",
        -52407 => "TOO_MANY_CLIENT",
        -52405 => "TOO_MANY_REQUEST",
        -52435 => "VOD_SESSION_FULL",
        -52417 => "VOD_SESSION_OCCUPIED",
        -40404 => "DEVICE_BLOCKED",
        -40211 => "MISSING_NECESSARY_PARAMS",
        -40413 => "INVALID_NONCE",
        _ => "unknown error",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CamRecording {
    pub start: i64,
    pub end: i64,
    pub video_type: i64,
}

pub struct Tapo {
    transport: Box<dyn Transport>,
    pub host: String,
    pub media_port: u16,
    pub cloud_password: String,
    pub player_id: String,
    user_id: Option<Value>,
}

impl Tapo {
    /// Connects and picks the transport like pytapo does: TPAP when the camera
    /// advertises it (or refuses the legacy login), the classic one otherwise.
    pub fn connect(host: &str, control_port: u16, media_port: u16, cloud_password: &str, transport: &str) -> Result<Self> {
        let mut probe = https::HttpsClient::new(host, control_port, CONNECTION_TIMEOUT)?;
        let kind = match transport {
            "auto" | "" => {
                if let Some(info) = tpap::discover(&mut probe) {
                    Some(info)
                } else if tpap::rejects_legacy_login(&mut probe) {
                    Some(json!({}))
                } else {
                    None
                }
            }
            "tpap" => Some(tpap::discover(&mut probe).unwrap_or_else(|| json!({}))),
            "classic" => None,
            other => bail!("unknown camera.transport {other:?} (auto, classic, tpap)"),
        };
        let http = https::HttpsClient::new(host, control_port, CONNECTION_TIMEOUT)?;
        drop(probe);
        let transport: Box<dyn Transport> = match kind {
            Some(info) => Box::new(tpap::Tpap::new(http, cloud_password, Some(info))),
            // pytapo's download example logs in as "admin" with the cloud password.
            None => Box::new(classic::Classic::new(http, "admin", cloud_password)),
        };
        let mut t = Self {
            transport,
            host: host.to_string(),
            media_port,
            cloud_password: cloud_password.to_string(),
            player_id: uuid::Uuid::new_v4().to_string(),
            user_id: None,
        };
        // Fail early with a clear message if the credentials are wrong.
        t.device_info().context("cannot talk to the camera")?;
        log::info!("connected to camera {} using the {} transport", host, t.transport.name());
        Ok(t)
    }

    pub fn transport_name(&self) -> &'static str {
        self.transport.name()
    }

    pub fn perform(&mut self, request: &Value) -> Result<Value> {
        let mut r = self.transport.send(request)?;
        let mut code = r.get("error_code").and_then(Value::as_i64).unwrap_or(0);
        if code == -40401 || code == -1 {
            r = self.transport.send(request)?;
            code = r.get("error_code").and_then(Value::as_i64).unwrap_or(0);
        }
        if code != 0 {
            bail!(TapoError::Camera { code, msg: error_message(code).to_string() });
        }
        Ok(r)
    }

    /// pytapo's `executeFunction`: one method wrapped in a multipleRequest.
    pub fn execute(&mut self, method: &str, params: Value) -> Result<Value> {
        let req = json!({"method": "multipleRequest", "params": {"requests": [{"method": method, "params": params}]}});
        let r = self.perform(&req)?;
        let data = r.pointer("/result/responses/0").cloned().ok_or_else(|| anyhow!("unexpected response to {method}: {r}"))?;
        let code = data.get("error_code").and_then(Value::as_i64).unwrap_or(0);
        if code != 0 {
            let msg = data.get("err_msg").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| error_message(code).to_string());
            bail!(TapoError::Camera { code, msg });
        }
        Ok(data.get("result").cloned().unwrap_or(data))
    }

    pub fn device_info(&mut self) -> Result<Value> {
        self.execute("getDeviceInfo", json!({"device_info": {"name": ["basic_info"]}}))
    }

    /// Camera clock (seconds since epoch as the camera sees it).
    pub fn clock(&mut self) -> Result<i64> {
        let r = self.execute("getClockStatus", json!({"system": {"name": "clock_status"}}))?;
        r.pointer("/system/clock_status/seconds_from_1970")
            .or_else(|| r.get("timestamp"))
            .and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
            .ok_or_else(|| anyhow!("camera did not report its clock: {r}"))
    }

    /// Seconds to add to camera timestamps to get real unix time.
    pub fn time_correction(&mut self) -> Result<i64> {
        let cam = self.clock()?;
        Ok(crate::timeutil::now_ts() - cam)
    }

    pub fn user_id(&mut self) -> Result<Value> {
        if let Some(u) = &self.user_id {
            return Ok(u.clone());
        }
        let r = match self.execute("getUserID", json!({"system": {"get_user_id": "null"}})) {
            // happens e.g. while another download is running
            Err(e) if matches!(e.downcast_ref::<TapoError>(), Some(TapoError::Camera { code: -71101, .. })) => {
                self.execute("getUserID", json!({"system": {"get_user_id": "null"}}))?
            }
            r => r?,
        };
        let id = r.get("user_id").cloned().ok_or_else(|| anyhow!("camera returned no user_id"))?;
        self.user_id = Some(id.clone());
        Ok(id)
    }

    /// Days (YYYYMMDD) with recordings between the two dates.
    pub fn dates_with_video(&mut self, start: &str, end: &str) -> Result<Vec<String>> {
        let r = self.execute(
            "searchDateWithVideo",
            json!({"playback": {"search_year_utility": {"channel": [0], "end_date": end, "start_date": start}}}),
        )?;
        let results = r.pointer("/playback/search_results").ok_or_else(|| anyhow!("video playback is not supported by this camera"))?;
        let mut dates = Vec::new();
        collect_values(results, "date", &mut dates);
        dates.sort();
        dates.dedup();
        Ok(dates)
    }

    /// Recordings of one day (YYYYMMDD, camera local date).
    pub fn recordings_of_day(&mut self, date: &str) -> Result<Vec<CamRecording>> {
        for attempt in 0..2 {
            let id = self.user_id()?;
            let res = self.execute(
                "searchVideoOfDay",
                json!({"playback": {"search_video_utility": {
                    "channel": 0, "date": date, "end_index": 999999999, "id": id, "start_index": 0
                }}}),
            );
            match res {
                Ok(r) => {
                    let results = r
                        .pointer("/playback/search_video_results")
                        .ok_or_else(|| anyhow!("video playback is not supported by this camera"))?;
                    let mut out = Vec::new();
                    collect_recordings(results, &mut out);
                    out.sort_by_key(|r| (r.start, r.end));
                    out.dedup();
                    return Ok(out);
                }
                Err(e) => {
                    // user ID expired, get a new one
                    let stale = matches!(e.downcast_ref::<TapoError>(), Some(TapoError::Camera { code: -71103 | -71105, .. }));
                    if attempt == 0 && stale {
                        self.user_id = None;
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        unreachable!()
    }
}

fn as_i64(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
}

fn collect_values(v: &Value, key: &str, out: &mut Vec<String>) {
    match v {
        Value::Array(a) => a.iter().for_each(|x| collect_values(x, key, out)),
        Value::Object(m) => {
            if let Some(d) = m.get(key) {
                match d {
                    Value::String(s) => out.push(s.clone()),
                    Value::Number(n) => out.push(n.to_string()),
                    _ => {}
                }
            } else {
                m.values().for_each(|x| collect_values(x, key, out));
            }
        }
        _ => {}
    }
}

/// The result is a list of single-key objects (`{"video_0": {"startTime": ..}}`);
/// accept any nesting that ends in an object with startTime/endTime.
fn collect_recordings(v: &Value, out: &mut Vec<CamRecording>) {
    match v {
        Value::Array(a) => a.iter().for_each(|x| collect_recordings(x, out)),
        Value::Object(m) => {
            let start = m.get("startTime").or_else(|| m.get("start_time")).and_then(as_i64);
            let end = m.get("endTime").or_else(|| m.get("end_time")).and_then(as_i64);
            if let (Some(start), Some(end)) = (start, end) {
                let video_type = m.get("vedio_type").or_else(|| m.get("video_type")).and_then(as_i64).unwrap_or(0);
                if end > start {
                    out.push(CamRecording { start, end, video_type });
                }
            } else {
                m.values().for_each(|x| collect_recordings(x, out));
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_recording_lists() {
        let v = json!([
            {"search_video_results_0": {"startTime": 1700000000, "endTime": 1700000600, "vedio_type": 2}},
            {"search_video_results_1": {"startTime": "1700000700", "endTime": "1700001000", "vedio_type": 1}}
        ]);
        let mut out = Vec::new();
        collect_recordings(&v, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1], CamRecording { start: 1700000700, end: 1700001000, video_type: 1 });
        let mut dates = Vec::new();
        collect_values(&json!([{"search_results_0": {"date": "20261012"}}, {"date": "20261013"}]), "date", &mut dates);
        assert_eq!(dates, vec!["20261012", "20261013"]);
    }
}
