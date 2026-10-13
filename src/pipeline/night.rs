//! Turns one day of recordings (a night and the naps after it) into one
//! timelapse segment.

use super::analyzer::Analyzer;
use super::select::{FrameInfo, Part, Selector, SelectorConfig};
use super::Ctx;
use crate::db::{FrameRow, NightStats, Recording, Variant};
use crate::timeutil::{age_label, date_label, weekday_label};
use crate::video::ffmpeg::{self, DecodeOptions, Decoder, Encoder};
use crate::vision::image::Yuv;
use anyhow::{Context, Result};
use chrono::NaiveDate;
use crossbeam_channel::bounded;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::Ordering;

/// A finished segment file and the frames in it.
pub type NewSegment = (PathBuf, Vec<FrameRow>);

pub struct NightOutput {
    /// Per timelapse version: the new segment (None = no usable frames).
    pub segments: Vec<(Variant, Option<NewSegment>)>,
    pub stats: NightStats,
}

fn caption(ctx: &Ctx, night: NaiveDate, t: f64, part: Part) -> String {
    let o = &ctx.cfg.overlay;
    let local = ctx.clock.zone.to_local(t as i64);
    // a night is labelled with its evening's date, a nap with its own day
    let (mut s, date) = match part {
        Part::Nap => (o.nap_template.clone(), local.date()),
        _ => (o.template.clone(), night),
    };
    let n = (night - ctx.night_one).num_days() + 1;
    s = s.replace("{night}", &n.to_string());
    s = s.replace("{date}", &date_label(date, &o.language));
    s = s.replace("{weekday}", weekday_label(date, &o.language));
    s = s.replace("{time}", &local.format("%H:%M").to_string());
    let age = ctx.birth.map(|b| age_label(b, date, &o.language)).unwrap_or_default();
    s = s.replace("{age}", &age);
    // tidy separators left by empty placeholders
    let s = s.replace("·  ·", "·");
    s.trim().trim_end_matches('·').trim().trim_start_matches('·').trim().to_string()
}

/// A piece of a recording to decode.
struct Piece {
    rec: Recording,
    /// offset into the file and length, seconds
    offset: f64,
    length: f64,
    /// absolute start (unix time)
    start: i64,
    part: Part,
}

fn plan(recs: &[Recording], ranges: &[(i64, i64, Part)]) -> Vec<Piece> {
    let mut out = Vec::new();
    for r in recs {
        for &(a, b, part) in ranges {
            let s = r.start_ts.max(a);
            let e = r.end_ts.min(b);
            if e > s {
                out.push(Piece { rec: r.clone(), offset: (s - r.start_ts) as f64, length: (e - s) as f64, start: s, part });
            }
        }
    }
    out.sort_by_key(|p| p.start);
    out
}

/// One output of the day: the regular timelapse or the uncensored one.
struct Track {
    variant: Variant,
    selector: Selector<Rc<Vec<u8>>>,
    tmp: PathBuf,
    encoder: Option<Encoder>,
    rows: Vec<FrameRow>,
}

type Best = Option<(f32, f64, Vec<u8>)>;

/// Captions, encodes and indexes the frames a selector released.
fn emit(ctx: &mut Ctx, night: NaiveDate, track: &mut Track, frames: Vec<(FrameInfo, Rc<Vec<u8>>)>, best: &mut Best) -> Result<()> {
    let (w, h) = (ctx.encoder.width as usize, ctx.encoder.height as usize);
    for (info, payload) in frames {
        // the other track may still hold the same frame
        let mut data = Rc::try_unwrap(payload).unwrap_or_else(|rc| (*rc).clone());
        if track.variant == Variant::Censored && best.as_ref().map(|(s, _, _)| info.score() > *s).unwrap_or(true) {
            *best = Some((info.score(), info.t, data.clone()));
        }
        let text = ctx.overlay.is_some().then(|| caption(ctx, night, info.t, info.part));
        if let (Some(o), Some(text)) = (ctx.overlay.as_mut(), text) {
            o.draw(&mut data, w, h, &text);
        }
        if track.encoder.is_none() {
            if track.variant == Variant::Uncensored {
                create_private(&track.tmp)?;
            }
            track.encoder = Some(Encoder::spawn(&ctx.encoder, &track.tmp)?);
        }
        track.encoder.as_mut().unwrap().write_frame(&data)?;
        track.rows.push(FrameRow {
            idx: track.rows.len() as i64,
            ts: info.t,
            motion: info.motion,
            private: info.private,
            presence: info.presence.unwrap_or(-1.0),
            interest: info.interest,
        });
    }
    Ok(())
}

/// Creates (or truncates) a file only its owner and group can read; ffmpeg
/// then writes into it keeping the mode.
pub fn create_private(path: &Path) -> Result<()> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o640)
        .open(path)
        .with_context(|| format!("cannot create {}", path.display()))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o640))?;
    Ok(())
}

/// `outputs`: the timelapse versions to build with their manual redactions;
/// the first one must be the censored timelapse (it provides the statistics).
pub fn process(
    ctx: &mut Ctx,
    night: NaiveDate,
    recs: &[Recording],
    outputs: Vec<(Variant, Vec<(f64, f64)>)>,
    segment_dir: &Path,
) -> Result<NightOutput> {
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
        nap_min: cfg.schedule.nap_min_minutes * 60.0,
        nap_gap: cfg.schedule.nap_gap_minutes * 60.0,
        privacy: true,
    };
    anyhow::ensure!(outputs.first().map(|o| o.0) == Some(Variant::Censored), "the censored timelapse comes first");
    let mut tracks: Vec<Track> = outputs
        .into_iter()
        .map(|(variant, blocks)| {
            let (sc, suffix) = match variant {
                Variant::Censored => (sel_cfg.clone(), ""),
                Variant::Uncensored => (sel_cfg.uncensored(), "-uncensored"),
            };
            Track {
                variant,
                selector: Selector::new(sc, blocks),
                tmp: segment_dir.join(format!("{night}{suffix}.mp4.tmp")),
                encoder: None,
                rows: Vec::new(),
            }
        })
        .collect();
    let pieces = plan(recs, &ctx.clock.plan(night));
    let total: f64 = pieces.iter().map(|p| p.length).sum();
    let mut best: Best = None;
    let mut stats = NightStats::default();

    let pb = ProgressBar::new(total.max(1.0) as u64);
    pb.set_style(ProgressStyle::with_template("  night {msg} [{bar:30}] {percent:>3}% {elapsed} eta {eta}").unwrap().progress_chars("=> "));
    pb.set_message(night.to_string());

    let mut last_t = f64::NEG_INFINITY;
    let mut done = 0.0;
    let mut models = std::mem::replace(&mut ctx.models, super::analyzer::Models { nudenet: None, clip: None });
    let result = (|| -> Result<()> {
        let mut analyzer = Analyzer::new(&mut models, &cfg);
        for piece in &pieces {
            if ctx.stop.load(Ordering::Relaxed) {
                anyhow::bail!("interrupted");
            }
            let path = PathBuf::from(piece.rec.file.as_deref().unwrap_or_default());
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
                start: piece.offset,
                length: Some(piece.length),
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
                let t = piece.start as f64 + i as f64 / afps;
                i += 1;
                if t <= last_t {
                    continue; // overlapping recordings
                }
                last_t = t;
                let frame = Rc::new(frame);
                let mut info = analyzer.analyze(t, &Yuv::new(&frame, w, h))?;
                info.part = piece.part;
                // every output sees the same analysed frame; the picture is shared
                for track in tracks.iter_mut() {
                    let out = track.selector.push(info.clone(), || frame.clone());
                    emit(ctx, night, track, out, &mut best)?;
                }
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
            done += piece.length;
            pb.set_position(done as u64);
        }
        for track in tracks.iter_mut() {
            let out = track.selector.finish();
            emit(ctx, night, track, out, &mut best)?;
        }
        log::debug!("detector runs: nudity {}, scene {}", analyzer.nsfw_runs, analyzer.clip_runs);
        Ok(())
    })();
    ctx.models = models;
    pb.finish_and_clear();
    if let Err(e) = result {
        for t in tracks {
            drop(t.encoder);
            let _ = std::fs::remove_file(&t.tmp);
        }
        return Err(e);
    }

    let s = &tracks[0].selector.stats;
    stats.analyzed = s.analyzed as i64;
    stats.emitted = tracks[0].rows.len() as i64;
    stats.skipped_private = (s.private + s.dropped_private) as i64;
    stats.skipped_idle = s.idle as i64;
    stats.skipped_dark = s.dark as i64;
    stats.nap_frames = s.emitted_nap as i64;
    stats.skipped_short = s.dropped_short as i64;
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
    // unique names: the previous version of a reprocessed night stays valid until the index is updated
    let stamp = crate::timeutil::now_ts();
    let mut segments = Vec::new();
    for t in tracks {
        let segment = match t.encoder {
            Some(enc) => {
                enc.finish().with_context(|| format!("encoding the night's {} segment failed", t.variant.key()))?;
                let suffix = if t.variant == Variant::Uncensored { "-uncensored" } else { "" };
                let final_path = segment_dir.join(format!("{night}-{stamp}{suffix}.mp4"));
                std::fs::rename(&t.tmp, &final_path)?;
                Some((final_path, t.rows))
            }
            None => None,
        };
        segments.push((t.variant, segment));
    }
    Ok(NightOutput { segments, stats })
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
        let p = plan(&[rec(0, 100), rec(150, 400)], &[(50, 200, Part::Night), (300, 350, Part::Nap)]);
        let v: Vec<(f64, f64, i64, Part)> = p.iter().map(|x| (x.offset, x.length, x.start, x.part)).collect();
        assert_eq!(v, vec![(50.0, 50.0, 50, Part::Night), (0.0, 50.0, 150, Part::Night), (150.0, 50.0, 300, Part::Nap)]);
    }
}
