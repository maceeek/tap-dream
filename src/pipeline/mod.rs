//! Orchestration: which nights are ready, processing them into segments,
//! appending segments to the master timelapse, redactions and rebuilds.

pub mod analyzer;
pub mod night;
pub mod select;

use crate::config::Config;
use crate::db::{Db, FrameRow, MasterState, NightStats, RecStatus, Recording, Variant};
use crate::timeutil::{fmt_ts, now_ts, parse_date, NightClock};
use crate::video::ffmpeg::{self, DecodeOptions, Decoder, Encoder, EncoderSpec, TIMESCALE};
use crate::video::fmp4::{self, Compat, MasterLayout, Segment};
use crate::video::overlay::{find_font, Overlay};
use analyzer::Models;
use anyhow::{anyhow, bail, Context, Result};
use chrono::NaiveDate;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct Ctx {
    pub cfg: Config,
    pub clock: NightClock,
    pub models: Models,
    pub overlay: Option<Overlay>,
    pub encoder: EncoderSpec,
    pub cuda_decode: bool,
    pub night_one: NaiveDate,
    pub birth: Option<NaiveDate>,
    pub stop: Arc<AtomicBool>,
}

pub fn segments_dir(cfg: &Config) -> PathBuf {
    cfg.general.data_dir.join("segments")
}

/// The output format is fixed by the first segment so all nights can be
/// joined without re-encoding. Changing it needs `rebuild --reencode`.
pub fn encoder_spec(cfg: &Config, db: &Db, caps: &ffmpeg::Caps) -> Result<EncoderSpec> {
    let t = &cfg.timelapse;
    let wanted = EncoderSpec::choose(caps, &t.codec, &t.encoder, t.quality, t.width, t.height, t.fps)?;
    if let Some(locked) = db.meta_get("encoder")? {
        let l: serde_json::Value = serde_json::from_str(&locked)?;
        let lw = l["width"].as_u64().unwrap_or(0) as u32;
        let lh = l["height"].as_u64().unwrap_or(0) as u32;
        let lf = l["fps"].as_u64().unwrap_or(0) as u32;
        let lcodec = l["codec"].as_str().unwrap_or("");
        if (lw, lh, lf) != (t.width, t.height, t.fps) || lcodec != t.codec {
            log::warn!(
                "timelapse format is locked to {lw}x{lh} @ {lf} fps ({lcodec}); config differs. Run `tapdream rebuild --reencode` to switch."
            );
            let mut s = EncoderSpec::choose(caps, lcodec, &t.encoder, t.quality, lw, lh, lf)?;
            s.quality = l["quality"].as_u64().unwrap_or(t.quality as u64) as u32;
            return Ok(s);
        }
    }
    Ok(wanted)
}

pub fn lock_encoder(db: &Db, cfg: &Config, spec: &EncoderSpec) -> Result<()> {
    let v = serde_json::json!({
        "width": spec.width, "height": spec.height, "fps": spec.fps,
        "codec": cfg.timelapse.codec, "quality": spec.quality, "encoder": spec.name,
    });
    db.meta_set("encoder", &v.to_string())
}

fn cuda_decode(cfg: &Config, caps: &ffmpeg::Caps) -> bool {
    match cfg.analysis.hwaccel.as_str() {
        "cuda" => true,
        "none" => false,
        _ => caps.nvidia && caps.hwaccels.contains("cuda"),
    }
}

fn first_night(cfg: &Config, db: &Db, fallback: NaiveDate) -> Result<NaiveDate> {
    if !cfg.overlay.start_date.is_empty() {
        return parse_date(&cfg.overlay.start_date);
    }
    if !cfg.overlay.birth_date.is_empty() {
        return parse_date(&cfg.overlay.birth_date);
    }
    if let Some(n) = db.first_night()? {
        return parse_date(&n);
    }
    if let Some(n) = db.meta_get("first_night")? {
        return parse_date(&n);
    }
    db.meta_set("first_night", &fallback.to_string())?;
    Ok(fallback)
}

pub fn build_ctx(cfg: &Config, db: &Db, clock: &NightClock, stop: Arc<AtomicBool>, first: NaiveDate) -> Result<Ctx> {
    let caps = ffmpeg::caps()?;
    let encoder = encoder_spec(cfg, db, &caps)?;
    let overlay = if cfg.overlay.enabled {
        match find_font(&cfg.overlay.font) {
            Some(f) => Some(Overlay::new(&f, encoder.height, cfg.overlay.size, &cfg.overlay.position)?),
            None => {
                log::warn!("no font found for the caption (set overlay.font); captions disabled");
                None
            }
        }
    } else {
        None
    };
    let models = Models::load(cfg)?;
    let cuda = cuda_decode(cfg, &caps);
    log::info!(
        "encoder {} ({}x{} @ {} fps), decoding on {}",
        encoder.name,
        encoder.width,
        encoder.height,
        encoder.fps,
        if cuda { "GPU (NVDEC)" } else { "CPU" }
    );
    Ok(Ctx {
        cfg: cfg.clone(),
        clock: clock.clone(),
        models,
        overlay,
        encoder,
        cuda_decode: cuda,
        night_one: first_night(cfg, db, first)?,
        birth: if cfg.overlay.birth_date.is_empty() { None } else { Some(parse_date(&cfg.overlay.birth_date)?) },
        stop,
    })
}

/// Days that are over, fully downloaded and not processed yet.
pub fn ready_nights(cfg: &Config, db: &Db, clock: &NightClock) -> Result<Vec<NaiveDate>> {
    let now = now_ts();
    let discovered_until: i64 = db.meta_get("last_discovery")?.and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut candidates = BTreeSet::new();
    for r in db.recordings_with_status(RecStatus::Downloaded)? {
        for n in clock.nights_touching(r.start_ts, r.end_ts) {
            candidates.insert(n);
        }
    }
    let mut out = Vec::new();
    'nights: for night in candidates {
        let key = night.to_string();
        if db.night_status(&key)?.as_deref() == Some("processed") {
            continue;
        }
        // the plan includes the privacy context just after the day
        let end = clock.plan_end(night);
        let settle = end + cfg.schedule.grace_minutes * 60;
        if now < settle || discovered_until < end + cfg.camera.min_age_secs {
            continue; // not over yet, or the camera was not asked about its last hours
        }
        let plan = clock.plan(night);
        let (s, e) = (plan.first().map(|i| i.0).unwrap_or(0), plan.last().map(|i| i.1).unwrap_or(0));
        let mut usable = false;
        for r in db.recordings_overlapping(s, e)? {
            let overlaps = |p: &&(i64, i64, crate::timeutil::Part)| r.start_ts < p.1 && r.end_ts > p.0;
            if !plan.iter().any(|p| overlaps(&p)) {
                continue;
            }
            match r.status {
                RecStatus::New => continue 'nights, // still waiting for the download
                RecStatus::Downloaded => {
                    usable |= plan.iter().filter(|p| p.2 != crate::timeutil::Part::Context).any(|p| overlaps(&p));
                }
                _ => {}
            }
        }
        if usable {
            out.push(night);
        }
    }
    Ok(out)
}

fn blocks_for(db: &Db, variant: Variant) -> Result<Vec<(f64, f64)>> {
    Ok(db.redactions(variant)?.into_iter().map(|(a, b)| (a as f64, b as f64)).collect())
}

/// Processes every ready night (oldest first) and appends it to the timelapse.
pub fn process_ready(cfg: &Config, db: &Db, clock: &NightClock, stop: Arc<AtomicBool>) -> Result<usize> {
    let nights = ready_nights(cfg, db, clock)?;
    if nights.is_empty() {
        log::info!("no complete night waiting to be processed");
        update_master(cfg, db)?;
        return Ok(0);
    }
    let mut ctx = build_ctx(cfg, db, clock, stop.clone(), nights[0])?;
    let seg_dir = segments_dir(cfg);
    std::fs::create_dir_all(&seg_dir)?;
    let mut done = 0;
    for night in nights {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        if let Err(e) = process_one(&mut ctx, db, night, &seg_dir) {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            log::error!("night {night} failed: {e:#}");
            let (key, msg) = (night.to_string(), format!("{e:#}"));
            if db.night_status(&key)?.as_deref() == Some("reprocess") {
                // the previous version of the night stays in the timelapse
                db.set_night_status(&key, "processed", Some(&msg))?;
            } else {
                // Keep going with the other nights; this one is retried next run.
                db.set_night(&key, "error", None, None, &NightStats::default(), now_ts(), Some(&msg))?;
            }
            continue;
        }
        done += 1;
        update_master(cfg, db)?;
        purge_downloads(cfg, db, clock)?;
    }
    Ok(done)
}

pub fn process_one(ctx: &mut Ctx, db: &Db, night: NaiveDate, seg_dir: &Path) -> Result<()> {
    let clock = ctx.clock.clone();
    let plan = clock.plan(night);
    let (s, e) = (plan.first().map(|p| p.0).unwrap_or(0), plan.last().map(|p| p.1).unwrap_or(0));
    let recs: Vec<_> = db
        .recordings_overlapping(s, e)?
        .into_iter()
        .filter(|r| r.status == RecStatus::Downloaded && r.file.as_deref().map(|f| Path::new(f).exists()).unwrap_or(false))
        .collect();
    log::info!("processing day {night} ({} recordings)", recs.len());
    let started = std::time::Instant::now();
    let outputs = ctx.cfg.variants().into_iter().map(|v| Ok((v, blocks_for(db, v)?))).collect::<Result<Vec<_>>>()?;
    let out = night::process(ctx, night, &recs, outputs, seg_dir)?;
    let st = &out.stats;
    let afps = ctx.cfg.analysis.fps;
    let mins = |n: i64| n as f64 / afps / 60.0;
    log::info!(
        "day {night}: {} frames kept ({} from naps) from {:.0} min analysed; skipped {:.0} min private, {:.0} min empty/static, {:.0} min dark, {} frames of short daytime visits ({:.0} s)",
        st.emitted,
        st.nap_frames,
        mins(st.analyzed),
        mins(st.skipped_private),
        mins(st.skipped_idle),
        mins(st.skipped_dark),
        st.skipped_short,
        started.elapsed().as_secs_f64()
    );
    let now = now_ts();
    let mut ids = [None, None];
    if !ctx.cfg.timelapse.uncensored {
        // not built now: a night processed again keeps its uncensored version
        ids[1] = db.uncensored_segment_of(&night.to_string())?;
    }
    for (variant, segment) in out.segments {
        let Some((path, rows)) = segment else { continue };
        if db.meta_get("encoder")?.is_none() {
            lock_encoder(db, &ctx.cfg, &ctx.encoder)?;
        }
        let duration = rows.len() as i64 * (TIMESCALE / ctx.encoder.fps) as i64;
        let id = db.insert_segment(&night.to_string(), variant, &path.to_string_lossy(), &rows, duration, now)?;
        ids[(variant == Variant::Uncensored) as usize] = Some(id);
        if variant == Variant::Uncensored {
            log::info!("day {night}: {} frames in the uncensored timelapse", rows.len());
        }
    }
    let note = ids[0].is_none().then_some("no usable frames");
    db.set_night(&night.to_string(), "processed", ids[0], ids[1], st, now, note)?;
    // segments of earlier runs of this night are no longer referenced
    let mut live = Vec::new();
    for v in [Variant::Censored, Variant::Uncensored] {
        live.extend(db.active_segments(v)?.into_iter().map(|s| s.file));
    }
    for old in db.segments_unreferenced()? {
        if !live.contains(&old.file) {
            let _ = std::fs::remove_file(&old.file);
        }
        db.delete_segment(old.id)?;
    }
    Ok(())
}

/// Deletes downloaded recordings no unprocessed night needs any more.
pub fn purge_downloads(cfg: &Config, db: &Db, clock: &NightClock) -> Result<()> {
    if cfg.schedule.keep_downloads {
        return Ok(());
    }
    for r in db.recordings_with_status(RecStatus::Downloaded)? {
        if !crate::sync::is_needed(db, clock, &r)? {
            if let Some(f) = &r.file {
                let _ = std::fs::remove_file(f);
            }
            db.mark_purged(r.id)?;
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ master file

fn layout_of(m: &MasterState) -> MasterLayout {
    MasterLayout {
        data_end: m.data_end,
        total_duration: m.total_duration,
        next_seq: m.next_seq,
        timescale: m.timescale,
        movie_timescale: m.movie_timescale,
        track_id: m.track_id,
        stsd: m.stsd.clone(),
        moov_offset: m.moov_offset,
        moov_size: m.moov_size,
    }
}

fn state_of(path: &Path, l: &MasterLayout, segments: Vec<i64>) -> MasterState {
    MasterState {
        path: path.to_string_lossy().to_string(),
        data_end: l.data_end,
        total_duration: l.total_duration,
        next_seq: l.next_seq,
        timescale: l.timescale,
        track_id: l.track_id,
        stsd: l.stsd.clone(),
        moov_offset: l.moov_offset,
        moov_size: l.moov_size,
        movie_timescale: l.movie_timescale,
        segments,
    }
}

fn check_compat(layout: &MasterLayout, s: &Segment, name: &str) -> Result<()> {
    match fmp4::compat(layout, &s.info) {
        Compat::Same => Ok(()),
        Compat::InBand => {
            log::warn!("segment {name} has different codec parameters (encoder changed?); relying on in-band parameter sets");
            Ok(())
        }
        Compat::Incompatible(why) => bail!("segment {name} cannot be joined to the timelapse ({why}); run `tapdream rebuild --reencode`"),
    }
}

fn label(variant: Variant) -> &'static str {
    match variant {
        Variant::Censored => "timelapse",
        Variant::Uncensored => "uncensored timelapse",
    }
}

/// Appends new segments to every timelapse that is being built (or rebuilds it when needed).
pub fn update_master(cfg: &Config, db: &Db) -> Result<()> {
    for v in cfg.variants() {
        update_variant(cfg, db, v)?;
    }
    Ok(())
}

fn update_variant(cfg: &Config, db: &Db, variant: Variant) -> Result<()> {
    let path = cfg.output_path(variant);
    let segs = db.active_segments(variant)?;
    if segs.is_empty() {
        // e.g. the only night was redacted completely: nothing may stay behind
        return remove_master(&path, db, variant);
    }
    let ids: Vec<i64> = segs.iter().map(|s| s.id).collect();
    let Some(mut state) = db.master(variant)? else {
        return rebuild_variant(cfg, db, variant);
    };
    if !path.exists() || Path::new(&state.path) != path || !ids.starts_with(&state.segments) {
        log::info!("{} needs a rebuild (segments changed or file moved)", label(variant));
        return rebuild_variant(cfg, db, variant);
    }
    let mut layout = layout_of(&state);
    let mut index = db.master_fragments(variant)?;
    if fmp4::recover(&path, &layout, &index)? {
        log::warn!("removed an incomplete append from the {} (interrupted run)", label(variant));
    }
    for seg in &segs[state.segments.len()..] {
        let s = Segment::load(Path::new(&seg.file))?;
        check_compat(&layout, &s, &seg.night)?;
        let new = fmp4::append_segment(&path, &mut layout, &s, &mut index)?;
        state.segments.push(seg.id);
        let st = state_of(&path, &layout, state.segments.clone());
        db.commit_master(variant, &st, &new, false)?;
        state = st;
        log::info!(
            "appended night {} ({} frames) – {} is now {}",
            seg.night,
            seg.frames,
            label(variant),
            human_duration(layout.total_duration as f64 / layout.timescale as f64)
        );
    }
    Ok(())
}

/// Writes every timelapse that is being built again from its segments.
pub fn rebuild_master(cfg: &Config, db: &Db) -> Result<()> {
    for v in cfg.variants() {
        rebuild_variant(cfg, db, v)?;
    }
    Ok(())
}

fn rebuild_variant(cfg: &Config, db: &Db, variant: Variant) -> Result<()> {
    let path = cfg.output_path(variant);
    let segs = db.active_segments(variant)?;
    if segs.is_empty() {
        log::info!("there are no processed nights with footage for the {}", label(variant));
        return remove_master(&path, db, variant);
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("mp4.rebuild");
    if variant == Variant::Uncensored {
        night::create_private(&tmp)?;
    }
    let first = Segment::load(Path::new(&segs[0].file))?;
    let mut layout = fmp4::create_master(&tmp, &first)?;
    drop(first);
    let mut index = Vec::new();
    let mut all_new = Vec::new();
    for seg in &segs {
        let s = Segment::load(Path::new(&seg.file)).with_context(|| format!("segment of night {}", seg.night))?;
        check_compat(&layout, &s, &seg.night)?;
        all_new.extend(fmp4::append_segment(&tmp, &mut layout, &s, &mut index)?);
    }
    std::fs::rename(&tmp, &path)?;
    let st = state_of(&path, &layout, segs.iter().map(|s| s.id).collect());
    db.commit_master(variant, &st, &all_new, true)?;
    log::info!(
        "{} rebuilt: {} nights, {}",
        label(variant),
        segs.len(),
        human_duration(layout.total_duration as f64 / layout.timescale as f64)
    );
    Ok(())
}

fn remove_master(path: &Path, db: &Db, variant: Variant) -> Result<()> {
    if db.master(variant)?.is_some() || path.exists() {
        if path.exists() {
            std::fs::remove_file(path).with_context(|| format!("cannot remove {}", path.display()))?;
        }
        db.clear_master(variant)?;
        log::info!("{} removed (no footage left)", label(variant));
    }
    Ok(())
}

pub fn human_duration(secs: f64) -> String {
    let s = secs.round() as i64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

// ------------------------------------------------------------------ re-encoding

/// Re-encodes a segment keeping only the frames for which `keep` is true.
/// `source_fps` is the frame rate the segment was encoded with (frames map 1:1 to rows).
fn reencode_segment(
    spec: &EncoderSpec,
    source_fps: u32,
    input: &Path,
    output: &Path,
    frames: &[FrameRow],
    keep: impl Fn(&FrameRow) -> bool,
) -> Result<Vec<FrameRow>> {
    let opts = DecodeOptions {
        width: spec.width,
        height: spec.height,
        fps: source_fps as f64,
        crop: String::new(),
        cuda: false,
        keyframes_only: false,
        start: 0.0,
        length: None,
        still: false,
    };
    let mut dec = Decoder::spawn(input, &opts)?;
    let mut enc: Option<Encoder> = None;
    let mut kept = Vec::new();
    let mut buf = Vec::new();
    let mut i = 0usize;
    while dec.read_frame(&mut buf)? {
        if let Some(row) = frames.get(i) {
            if keep(row) {
                if enc.is_none() {
                    enc = Some(Encoder::spawn(spec, output)?);
                }
                enc.as_mut().unwrap().write_frame(&buf)?;
                let mut r = row.clone();
                r.idx = kept.len() as i64;
                kept.push(r);
            }
        }
        i += 1;
    }
    dec.finish()?;
    if i != frames.len() {
        log::warn!("segment {} has {i} frames but the index lists {}", input.display(), frames.len());
    }
    if let Some(e) = enc {
        e.finish()?;
    }
    Ok(kept)
}

/// Removes everything between two moments from the timelapse (and from any
/// night processed later). The uncensored timelapse is only touched when
/// `everywhere` is set. Returns the number of frames removed per version.
pub fn redact(cfg: &Config, db: &Db, from: i64, to: i64, everywhere: bool) -> Result<Vec<(Variant, usize)>> {
    if to <= from {
        bail!("--to must be after --from");
    }
    db.add_redaction(from, to, everywhere, now_ts())?;
    let caps = ffmpeg::caps()?;
    let spec = encoder_spec(cfg, db, &caps)?;
    let variants = if everywhere { vec![Variant::Censored, Variant::Uncensored] } else { vec![Variant::Censored] };
    let mut report = Vec::new();
    for variant in variants {
        let mut removed = 0;
        for seg in db.active_segments(variant)? {
            let frames = db.segment_frames(seg.id)?;
            let hit = |r: &FrameRow| r.ts >= from as f64 && r.ts <= to as f64;
            let n = frames.iter().filter(|r| hit(r)).count();
            if n == 0 {
                continue;
            }
            let input = PathBuf::from(&seg.file);
            let output = input.with_file_name(format!("{}-r{}{}.mp4", seg.night, now_ts(), suffix(variant)));
            if variant == Variant::Uncensored {
                night::create_private(&output)?;
            }
            let kept = reencode_segment(&spec, spec.fps, &input, &output, &frames, |r| !hit(r))?;
            let duration = kept.len() as i64 * (TIMESCALE / spec.fps) as i64;
            if kept.is_empty() {
                let _ = std::fs::remove_file(&output);
            }
            db.replace_segment_frames(seg.id, &output.to_string_lossy(), &kept, duration)?;
            let _ = std::fs::remove_file(&input);
            removed += n;
            log::info!("night {}: removed {n} frames from the {}", seg.night, label(variant));
        }
        if removed > 0 {
            refresh_variant(cfg, db, variant)?;
        }
        report.push((variant, removed));
    }
    Ok(report)
}

fn suffix(variant: Variant) -> &'static str {
    match variant {
        Variant::Censored => "",
        Variant::Uncensored => "-uncensored",
    }
}

/// Rebuilds a timelapse after its segments changed. A version that is not
/// being built any more but still exists on disk is rewritten too, so a
/// redaction never leaves the old content behind.
fn refresh_variant(cfg: &Config, db: &Db, variant: Variant) -> Result<()> {
    let built = cfg.variants().contains(&variant);
    let exists = db.master(variant)?.map(|m| Path::new(&m.path).exists()).unwrap_or(false);
    if built || exists {
        rebuild_variant(cfg, db, variant)
    } else {
        db.clear_master(variant)
    }
}

/// Re-encodes every segment with the current timelapse settings.
pub fn reencode_all(cfg: &Config, db: &Db) -> Result<()> {
    let caps = ffmpeg::caps()?;
    let t = &cfg.timelapse;
    let spec = EncoderSpec::choose(&caps, &t.codec, &t.encoder, t.quality, t.width, t.height, t.fps)?;
    let old_fps = db
        .meta_get("encoder")?
        .and_then(|v| serde_json::from_str::<serde_json::Value>(&v).ok())
        .and_then(|v| v["fps"].as_u64())
        .unwrap_or(t.fps as u64) as u32;
    for variant in [Variant::Censored, Variant::Uncensored] {
        for seg in db.active_segments(variant)? {
            let frames = db.segment_frames(seg.id)?;
            let input = PathBuf::from(&seg.file);
            let output = input.with_file_name(format!("{}-e{}{}.mp4", seg.night, now_ts(), suffix(variant)));
            if variant == Variant::Uncensored {
                night::create_private(&output)?;
            }
            // The old segment is decoded at its own size/rate and scaled to the new one.
            let kept = reencode_segment(&spec, old_fps, &input, &output, &frames, |_| true)?;
            let duration = kept.len() as i64 * (TIMESCALE / spec.fps) as i64;
            db.replace_segment_frames(seg.id, &output.to_string_lossy(), &kept, duration)?;
            let _ = std::fs::remove_file(&input);
            log::info!("re-encoded night {} ({})", seg.night, label(variant));
        }
    }
    lock_encoder(db, cfg, &spec)?;
    for variant in [Variant::Censored, Variant::Uncensored] {
        let existed = db.master(variant)?.is_some();
        db.clear_master(variant)?;
        if existed || cfg.variants().contains(&variant) {
            rebuild_variant(cfg, db, variant)?;
        }
    }
    Ok(())
}

/// Recordings a day's processing reads (its plan, privacy context included).
pub fn day_recordings(db: &Db, clock: &NightClock, night: NaiveDate) -> Result<Vec<Recording>> {
    let plan = clock.plan(night);
    let (s, e) = (plan.first().map(|p| p.0).unwrap_or(0), plan.last().map(|p| p.1).unwrap_or(0));
    Ok(db.recordings_overlapping(s, e)?.into_iter().filter(|r| plan.iter().any(|p| r.start_ts < p.1 && r.end_ts > p.0)).collect())
}

/// Recordings of a day that would have to be downloaded again to process it.
pub fn missing_recordings(db: &Db, clock: &NightClock, night: NaiveDate) -> Result<Vec<Recording>> {
    Ok(day_recordings(db, clock, night)?
        .into_iter()
        .filter(|r| match r.status {
            RecStatus::Purged | RecStatus::Skipped => true,
            RecStatus::Downloaded => !r.file.as_deref().map(|f| Path::new(f).exists()).unwrap_or(false),
            _ => false,
        })
        .collect())
}

/// Marks a night to be processed again. Its current version stays in the
/// timelapse until the new one replaces it (or for good if that fails). All
/// its recordings must be on disk (see `missing_recordings`).
pub fn reset_night(db: &Db, clock: &NightClock, night: NaiveDate) -> Result<()> {
    let missing = missing_recordings(db, clock, night)?.len();
    if missing > 0 {
        bail!("{missing} recordings of {night} are not on disk and the camera could not provide them again");
    }
    db.set_night_status(&night.to_string(), "reprocess", None)?;
    Ok(())
}

pub fn describe_night_stats(st: &NightStats, afps: f64) -> String {
    let m = |n: i64| n as f64 / afps / 60.0;
    format!(
        "{:>5} frames ({:>4} naps) | analysed {:>5.0} min | private {:>4.0} min | empty {:>5.0} min | dark {:>4.0} min",
        st.emitted,
        st.nap_frames,
        m(st.analyzed),
        m(st.skipped_private),
        m(st.skipped_idle),
        m(st.skipped_dark)
    )
}

pub fn parse_local(clock: &NightClock, s: &str) -> Result<i64> {
    let s = s.trim();
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"] {
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(clock.zone.timestamp_of(dt));
        }
    }
    Err(anyhow!("cannot parse {s:?}, expected \"YYYY-MM-DD HH:MM\""))
}

pub fn fmt(clock: &NightClock, ts: i64) -> String {
    fmt_ts(&clock.zone, ts)
}
