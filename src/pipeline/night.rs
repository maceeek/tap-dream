//! Turns one night of recordings into one timelapse segment.

use super::analyzer::Analyzer;
use super::select::{FrameInfo, Selector, SelectorConfig};
use super::Ctx;
use crate::db::{FrameRow, NightStats, Recording};
use crate::timeutil::{age_label, date_label, weekday_label};
use crate::video::ffmpeg::{self, DecodeOptions, Decoder, Encoder};
use crate::vision::image::Yuv;
use anyhow::{Context, Result};
use chrono::NaiveDate;
use crossbeam_channel::bounded;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

pub struct NightOutput {
    pub segment: Option<(PathBuf, Vec<FrameRow>)>,
    pub stats: NightStats,
}

fn caption(ctx: &Ctx, night: NaiveDate, t: f64) -> String {
    let o = &ctx.cfg.overlay;
    let mut s = o.template.clone();
    let n = (night - ctx.night_one).num_days() + 1;
    s = s.replace("{night}", &n.to_string());
    s = s.replace("{date}", &date_label(night, &o.language));
    s = s.replace("{weekday}", weekday_label(night, &o.language));
    let local = ctx.clock.zone.to_local(t as i64);
    s = s.replace("{time}", &local.format("%H:%M").to_string());
    // the whole night shows the same age (the evening's date)
    let age = ctx.birth.map(|b| age_label(b, night, &o.language)).unwrap_or_default();
    s = s.replace("{age}", &age);
    // tidy separators left by empty placeholders
    let s = s.replace("·  ·", "·");
    s.trim().trim_end_matches('·').trim().trim_start_matches('·').trim().to_string()
}

/// Pieces of recordings to decode: (recording, offset in file, length, absolute start).
fn plan(recs: &[Recording], intervals: &[(i64, i64)]) -> Vec<(Recording, f64, f64, i64)> {
    let mut out = Vec::new();
    for r in recs {
        for &(a, b) in intervals {
            let s = r.start_ts.max(a);
            let e = r.end_ts.min(b);
            if e > s {
                out.push((r.clone(), (s - r.start_ts) as f64, (e - s) as f64, s));
            }
        }
    }
    out.sort_by_key(|p| p.3);
    out
}

pub fn process(ctx: &mut Ctx, night: NaiveDate, recs: &[Recording], blocks: Vec<(f64, f64)>, segment_dir: &Path) -> Result<NightOutput> {
    let cfg = ctx.cfg.clone();
    let spec = ctx.encoder.clone();
    let (w, h) = (spec.width as usize, spec.height as usize);
    let afps = cfg.analysis.fps;
    let sel_cfg = SelectorConfig {
        analysis_fps: afps,
        interval: cfg.timelapse.interval_secs,
        calm_interval: cfg.timelapse.calm_interval_secs,
        idle_interval: cfg.timelapse.idle_interval_secs,
        motion_threshold: cfg.analysis.motion_threshold,
        activity_hold: cfg.analysis.activity_hold_secs,
        idle_after: cfg.analysis.idle_after_secs,
        presence_threshold: cfg.scene.presence_threshold,
        interest_weight: cfg.scene.interest_weight,
        before: cfg.privacy.before_secs,
        after: cfg.privacy.after_secs,
    };
    let mut selector: Selector<Vec<u8>> = Selector::new(sel_cfg, blocks);
    let pieces = plan(recs, &ctx.clock.capture_intervals(night));
    let total: f64 = pieces.iter().map(|p| p.2).sum();
    let tmp = segment_dir.join(format!("{night}.mp4.tmp"));
    let mut encoder: Option<Encoder> = None;
    let mut rows: Vec<FrameRow> = Vec::new();
    let mut best: Option<(f32, f64, Vec<u8>)> = None;
    let mut stats = NightStats::default();

    let pb = ProgressBar::new(total.max(1.0) as u64);
    pb.set_style(ProgressStyle::with_template("  night {msg} [{bar:30}] {percent:>3}% {elapsed} eta {eta}").unwrap().progress_chars("=> "));
    pb.set_message(night.to_string());

    let mut emit =
        |ctx: &mut Ctx, frames: Vec<(FrameInfo, Vec<u8>)>, encoder: &mut Option<Encoder>, rows: &mut Vec<FrameRow>| -> Result<()> {
            for (info, mut data) in frames {
                if best.as_ref().map(|(s, _, _)| info.score() > *s).unwrap_or(true) {
                    best = Some((info.score(), info.t, data.clone()));
                }
                let text = ctx.overlay.is_some().then(|| caption(ctx, night, info.t));
                if let (Some(o), Some(text)) = (ctx.overlay.as_mut(), text) {
                    o.draw(&mut data, w, h, &text);
                }
                if encoder.is_none() {
                    *encoder = Some(Encoder::spawn(&spec, &tmp)?);
                }
                encoder.as_mut().unwrap().write_frame(&data)?;
                rows.push(FrameRow {
                    idx: rows.len() as i64,
                    ts: info.t,
                    motion: info.motion,
                    private: info.private,
                    presence: info.presence.unwrap_or(-1.0),
                    interest: info.interest,
                });
            }
            Ok(())
        };

    let mut last_t = f64::NEG_INFINITY;
    let mut done = 0.0;
    let mut models = std::mem::replace(&mut ctx.models, super::analyzer::Models { nudenet: None, clip: None });
    let result = (|| -> Result<()> {
        let mut analyzer = Analyzer::new(&mut models, &cfg);
        for (rec, offset, length, abs_start) in &pieces {
            if ctx.stop.load(Ordering::Relaxed) {
                anyhow::bail!("interrupted");
            }
            let path = PathBuf::from(rec.file.as_deref().unwrap_or_default());
            let keyframes_only = match cfg.analysis.decode.as_str() {
                "keyframes" => true,
                "all" => false,
                _ => ffmpeg::keyframe_interval(&path).map(|k| k <= (2.0 / afps).max(2.0)).unwrap_or(false),
            };
            let opts = DecodeOptions {
                width: spec.width,
                height: spec.height,
                fps: afps,
                crop: cfg.timelapse.crop.clone(),
                cuda: ctx.cuda_decode,
                keyframes_only,
                start: *offset,
                length: Some(*length),
                still: false,
            };
            // A separate thread keeps ffmpeg busy while we analyse.
            let (tx, rx) = bounded::<Vec<u8>>(6);
            let reader_path = path.clone();
            let reader = std::thread::spawn(move || -> Result<(u64, bool)> {
                let mut opts = opts;
                loop {
                    let mut dec = Decoder::spawn(&reader_path, &opts)?;
                    loop {
                        let mut buf = Vec::new();
                        if !dec.read_frame(&mut buf)? {
                            break;
                        }
                        if tx.send(buf).is_err() {
                            break;
                        }
                    }
                    let n = dec.frames_read;
                    match dec.finish() {
                        // GPU decoding failed before producing anything: try the CPU
                        Err(e) if n == 0 && opts.cuda => {
                            log::warn!("GPU decoding failed ({e:#}); retrying on the CPU");
                            opts.cuda = false;
                        }
                        Err(e) => return Err(e),
                        Ok(()) => return Ok((n, opts.cuda)),
                    }
                }
            });
            analyzer.reset_motion();
            let mut i = 0u64;
            for frame in rx.iter() {
                let t = *abs_start as f64 + i as f64 / afps;
                i += 1;
                if t <= last_t {
                    continue; // overlapping recordings
                }
                last_t = t;
                let yuv = Yuv::new(&frame, w, h);
                let info = analyzer.analyze(t, &yuv)?;
                let out = selector.push(info, || frame.clone());
                emit(ctx, out, &mut encoder, &mut rows)?;
                if i.is_multiple_of(30) {
                    pb.set_position((done + i as f64 / afps) as u64);
                }
                if ctx.stop.load(Ordering::Relaxed) {
                    break;
                }
            }
            match reader.join().map_err(|_| anyhow::anyhow!("decoder thread panicked"))? {
                Ok((n, cuda)) => {
                    if n == 0 {
                        log::warn!("no frames decoded from {}", path.display());
                    }
                    if ctx.cuda_decode && !cuda {
                        ctx.cuda_decode = false; // don't try the GPU again this run
                    }
                }
                Err(e) => log::warn!("skipping unreadable recording {}: {e:#}", path.display()),
            }
            done += length;
            pb.set_position(done as u64);
        }
        let out = selector.finish();
        emit(ctx, out, &mut encoder, &mut rows)?;
        log::debug!("detector runs: nudity {}, scene {}", analyzer.nsfw_runs, analyzer.clip_runs);
        Ok(())
    })();
    ctx.models = models;
    pb.finish_and_clear();
    result?;

    let s = &selector.stats;
    stats.analyzed = s.analyzed as i64;
    stats.emitted = rows.len() as i64;
    stats.skipped_private = (s.private + s.dropped_private) as i64;
    stats.skipped_idle = s.idle as i64;
    stats.skipped_dark = s.dark as i64;
    if let Some((score, t, data)) = &best {
        stats.best_ts = Some(*t);
        stats.best_score = Some(*score as f64);
        if cfg.timelapse.highlights {
            let dir = cfg.general.data_dir.join("highlights");
            std::fs::create_dir_all(&dir)?;
            let path = dir.join(format!("{night}.jpg"));
            if let Err(e) = ffmpeg::write_jpeg(data, spec.width, spec.height, &path) {
                log::warn!("cannot save highlight: {e:#}");
            }
        }
    }
    let segment = match encoder {
        Some(enc) => {
            enc.finish().context("encoding the night's segment failed")?;
            // unique name: the previous version of a reprocessed night stays valid until the index is updated
            let final_path = segment_dir.join(format!("{night}-{}.mp4", crate::timeutil::now_ts()));
            std::fs::rename(&tmp, &final_path)?;
            Some((final_path, rows))
        }
        None => None,
    };
    Ok(NightOutput { segment, stats })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_clips_to_capture_window() {
        let rec = |s: i64, e: i64| Recording {
            id: 0,
            cam_start: s,
            cam_end: e,
            start_ts: s,
            end_ts: e,
            video_type: 0,
            status: crate::db::RecStatus::Downloaded,
            file: Some("x".into()),
            duration: None,
            attempts: 0,
            last_error: None,
        };
        let p = plan(&[rec(0, 100), rec(150, 400)], &[(50, 200), (300, 350)]);
        let v: Vec<(f64, f64, i64)> = p.iter().map(|x| (x.1, x.2, x.3)).collect();
        assert_eq!(v, vec![(50.0, 50.0, 50), (0.0, 50.0, 150), (150.0, 50.0, 300)]);
    }
}
