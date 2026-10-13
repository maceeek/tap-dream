//! Discovers recordings on the camera's SD card and downloads the ones that
//! are needed, keeping the index up to date.

use crate::config::Config;
use crate::db::{Db, RecStatus, Recording};
use crate::tapo::media::{self, Attempt, DownloadRequest, Method};
use crate::tapo::Tapo;
use crate::timeutil::{fmt_ts, now_ts, parse_date, NightClock};
use anyhow::{bail, Context, Result};
use chrono::Duration as ChronoDuration;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Debug, Default)]
pub struct SyncReport {
    pub listed: usize,
    pub new: usize,
    pub downloaded: usize,
    pub failed: usize,
    pub bytes: u64,
}

pub fn recordings_dir(cfg: &Config) -> PathBuf {
    cfg.general.data_dir.join("cache").join("recordings")
}

pub fn connect(cfg: &Config) -> Result<Tapo> {
    if cfg.camera.host.is_empty() {
        bail!("camera.host is not configured");
    }
    Tapo::connect(&cfg.camera.host, cfg.camera.control_port, cfg.camera.media_port, &cfg.cloud_password()?, &cfg.camera.transport)
}

/// Lists the camera's recordings of the configured history window into the index.
pub fn discover(cfg: &Config, db: &Db, clock: &NightClock, tapo: &mut Tapo, report: &mut SyncReport) -> Result<i64> {
    let correction = tapo.time_correction().unwrap_or_else(|e| {
        log::warn!("cannot read the camera clock ({e:#}), assuming it is correct");
        0
    });
    if correction.abs() > 120 {
        log::warn!("camera clock is off by {correction} s; recording times are corrected");
    }
    let today = clock.zone.to_local(now_ts()).date();
    let mut first = today - ChronoDuration::days(cfg.camera.history_days as i64);
    let since = if cfg.camera.since.is_empty() { None } else { Some(parse_date(&cfg.camera.since)?) };
    if let Some(s) = since {
        first = first.max(s - ChronoDuration::days(1));
    }
    let dates = tapo.dates_with_video(&first.format("%Y%m%d").to_string(), &today.format("%Y%m%d").to_string())?;
    let since_ts = since.map(|s| clock.night_range(s).0);
    let now = now_ts();
    for date in dates {
        let recs = tapo.recordings_of_day(&date).with_context(|| format!("cannot list recordings of {date}"))?;
        for r in recs {
            report.listed += 1;
            let (start, end) = (r.start + correction, r.end + correction);
            if since_ts.map(|s| end <= s).unwrap_or(false) {
                continue;
            }
            if db.upsert_recording(r.start, r.end, start, end, r.video_type, now)? {
                report.new += 1;
            }
        }
    }
    Ok(correction)
}

/// Does a day that is not processed yet use (or check for privacy) part of this recording?
pub fn is_needed(db: &Db, clock: &NightClock, rec: &Recording) -> Result<bool> {
    for night in clock.nights_touching(rec.start_ts, rec.end_ts) {
        if db.night_status(&night.to_string())?.as_deref() == Some("processed") {
            continue;
        }
        if clock.plan(night).iter().any(|&(a, b, _)| rec.start_ts < b && rec.end_ts > a) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn download_pending(
    cfg: &Config,
    db: &Db,
    clock: &NightClock,
    tapo: &mut Tapo,
    stop: &AtomicBool,
    report: &mut SyncReport,
) -> Result<()> {
    let dir = recordings_dir(cfg);
    fs::create_dir_all(&dir)?;
    let now = now_ts();
    let pending = db.recordings_with_status(RecStatus::New)?;
    let mut todo = Vec::new();
    for rec in pending {
        if !is_needed(db, clock, &rec)? {
            db.set_recording_status(rec.id, RecStatus::Skipped)?;
            continue;
        }
        if rec.end_ts > now - cfg.camera.min_age_secs {
            continue; // possibly still being recorded
        }
        todo.push(rec);
    }
    if todo.is_empty() {
        return Ok(());
    }
    let total: i64 = todo.iter().map(|r| r.end_ts - r.start_ts).sum();
    log::info!("downloading {} recordings ({:.1} h of video)", todo.len(), total as f64 / 3600.0);
    let user_id = tapo.user_id()?;
    for rec in todo {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        match download_one(cfg, tapo, &user_id, &rec, &dir, &clock.zone) {
            Ok((path, bytes, seconds)) => {
                db.mark_downloaded(rec.id, &path.to_string_lossy(), bytes, seconds, now_ts())?;
                report.downloaded += 1;
                report.bytes += bytes;
            }
            Err(e) => {
                let status = db.mark_attempt_failed(rec.id, &format!("{e:#}"), cfg.camera.max_attempts)?;
                report.failed += 1;
                log::warn!(
                    "download of {} failed{}: {e:#}",
                    fmt_ts(&clock.zone, rec.start_ts),
                    if status == RecStatus::Failed { " permanently" } else { " (will retry)" }
                );
            }
        }
    }
    Ok(())
}

fn download_one(
    cfg: &Config,
    tapo: &Tapo,
    user_id: &serde_json::Value,
    rec: &Recording,
    dir: &Path,
    zone: &crate::timeutil::Zone,
) -> Result<(PathBuf, u64, f64)> {
    let final_path = dir.join(format!("{}-{}.ts", rec.cam_start, rec.cam_end));
    let part_path = final_path.with_extension("ts.part");
    let length = (rec.cam_end - rec.cam_start) as f64;
    let mut method = if cfg.camera.download_method == "playback" { Method::Playback } else { Method::Download };
    let mut window = cfg.camera.window_size.max(1);
    let pb = ProgressBar::new((length * 10.0) as u64);
    pb.set_style(ProgressStyle::with_template("  {msg} [{bar:30}] {percent:>3}% {elapsed}").unwrap().progress_chars("=> "));
    pb.set_message(format!("{} ({:.0} s)", fmt_ts(zone, rec.start_ts), length));
    let mut attempt = 0;
    let result = loop {
        attempt += 1;
        let file = fs::File::create(&part_path)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        let req = DownloadRequest {
            host: &tapo.host,
            port: tapo.media_port,
            cloud_password: &tapo.cloud_password,
            user_id: user_id.clone(),
            player_id: &tapo.player_id,
            start: rec.cam_start,
            end: rec.cam_end,
            window,
            stall_timeout: Duration::from_secs(cfg.camera.stall_timeout_secs),
        };
        let r = media::stream_recording(&req, method, &mut out, &mut |s| pb.set_position((s * 10.0) as u64));
        drop(out);
        match r {
            Ok(Attempt::Unsupported) if method == Method::Download => {
                method = Method::Playback;
                attempt -= 1;
                continue;
            }
            Ok(Attempt::Unsupported) => bail!("camera refused the playback request"),
            Ok(Attempt::Done(d)) => {
                let complete = d.finished || d.seconds >= length - 5.0;
                if complete || attempt >= 2 {
                    if d.bytes == 0 {
                        bail!("camera sent no video");
                    }
                    if !complete {
                        if d.seconds < length * 0.5 {
                            bail!("only {:.0} of {:.0} s received", d.seconds, length);
                        }
                        log::warn!("recording {} is incomplete: {:.0} of {:.0} s", rec.cam_start, d.seconds, length);
                    }
                    break d;
                }
                log::debug!("download incomplete ({:.0}/{:.0} s), retrying with a smaller window", d.seconds, length);
                window = 50;
            }
            Err(e) if attempt < 2 => {
                log::debug!("download attempt failed: {e:#}; retrying");
                window = 50;
                std::thread::sleep(Duration::from_secs(2));
            }
            Err(e) => {
                let _ = fs::remove_file(&part_path);
                pb.finish_and_clear();
                return Err(e);
            }
        }
    };
    pb.finish_and_clear();
    fs::rename(&part_path, &final_path)?;
    log::info!(
        "downloaded {} ({:.0} s, {:.1} MB, {})",
        fmt_ts(zone, rec.start_ts),
        result.seconds,
        result.bytes as f64 / 1e6,
        if result.method == Method::Download { "download" } else { "playback" }
    );
    Ok((final_path, result.bytes, result.seconds))
}
