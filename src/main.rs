//! tapdream – one ever-growing, privacy-filtered timelapse from a Tapo camera.

mod config;
mod db;
mod models;
mod pipeline;
mod sync;
mod tapo;
mod timeutil;
mod video;
mod vision;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use config::Config;
use db::{Db, RecStatus, Variant};
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use timeutil::{fmt_ts, now_ts, parse_date, NightClock, Zone};

#[derive(Parser)]
#[command(name = "tapdream", version, about = "Long-term timelapse from Tapo camera recordings, with on-device privacy filtering")]
struct Cli {
    /// Configuration file
    #[arg(short, long, global = true, env = "TAPDREAM_CONFIG", default_value = "config.toml")]
    config: PathBuf,
    /// More output (-v debug, -vv trace)
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write an example configuration file
    Init {
        #[arg(default_value = "config.toml")]
        path: PathBuf,
    },
    /// Check ffmpeg, GPU, ONNX Runtime, models and (optionally) the camera
    Doctor {
        #[arg(long)]
        camera: bool,
    },
    /// Talk to the camera directly
    Camera {
        #[command(subcommand)]
        cmd: CameraCmd,
    },
    /// Download new recordings from the camera's SD card
    Sync,
    /// Turn finished nights into timelapse frames and append them
    Process,
    /// sync + process (what a timer/cron job should run)
    Run,
    /// Run forever: sync + process every schedule.daemon_interval_secs
    Daemon,
    /// Overview of the index and the timelapse
    Status,
    /// Per-night statistics
    Nights,
    /// Write the timelapse again from the per-night segments
    Rebuild {
        /// Re-encode all segments with the current [timelapse] settings
        #[arg(long)]
        reencode: bool,
    },
    /// Remove a time range from the timelapse (and from future processing)
    Redact {
        /// Local time, "YYYY-MM-DD HH:MM"
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: String,
        /// Remove it from the uncensored timelapse too
        #[arg(long)]
        also_uncensored: bool,
    },
    /// Process a day (or a range of days) again; deleted recordings are
    /// downloaded again if they are still on the camera's SD card
    Reprocess {
        /// First day, YYYY-MM-DD
        night: String,
        /// Last day (default: just the first one)
        #[arg(long)]
        to: Option<String>,
    },
    /// Write a regular (fast-start) MP4 copy of the timelapse
    Export {
        output: PathBuf,
        /// Export the uncensored timelapse
        #[arg(long)]
        uncensored: bool,
    },
    /// Show what the AI models think of an image or a video frame (for tuning thresholds)
    Inspect {
        file: PathBuf,
        /// Position in a video, seconds
        #[arg(long, default_value_t = 0.0)]
        at: f64,
    },
    /// AI models
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
}

#[derive(Subcommand)]
enum CameraCmd {
    /// Connection test and device info
    Info,
    /// List recordings of one day (YYYY-MM-DD, default today)
    List { date: Option<String> },
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// Download NudeNet and CLIP into data_dir/models
    Fetch {
        /// Skip the CLIP scene model (~600 MB)
        #[arg(long)]
        no_clip: bool,
    },
}

fn init_logging(verbose: u8) {
    let level = match verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(format!("tapdream={level},ort=warn")))
        .format_target(false)
        .format_timestamp_secs()
        .init();
}

struct App {
    cfg: Config,
    db: Db,
    clock: NightClock,
    _lock: Option<File>,
}

fn open_app(cli: &Cli, lock: bool) -> Result<App> {
    let cfg = Config::load(&cli.config).with_context(|| "use `tapdream init` to create a configuration")?;
    std::fs::create_dir_all(&cfg.general.data_dir).with_context(|| format!("cannot create {}", cfg.general.data_dir.display()))?;
    let lock_file = if lock {
        let f = File::create(cfg.general.data_dir.join(".lock"))?;
        if f.try_lock().is_err() {
            bail!("another tapdream process is working on {}", cfg.general.data_dir.display());
        }
        Some(f)
    } else {
        None
    };
    let db = Db::open(&cfg.general.data_dir.join("index.db"))?;
    let s = &cfg.schedule;
    let clock = NightClock::new(Zone::parse(&cfg.general.timezone)?, &s.day_boundary, &s.night_hours)?
        .with_naps(s.naps, &s.nap_hours)?
        .with_privacy_margins(cfg.privacy.before_secs, cfg.privacy.after_secs);
    Ok(App { cfg, db, clock, _lock: lock_file })
}

fn stop_flag() -> Arc<AtomicBool> {
    let stop = Arc::new(AtomicBool::new(false));
    let s = stop.clone();
    let _ = ctrlc::set_handler(move || {
        if s.swap(true, Ordering::SeqCst) {
            eprintln!("\nforced exit");
            std::process::exit(130);
        }
        eprintln!("\nstopping after the current step (Ctrl+C again to abort)…");
    });
    stop
}

fn do_sync(app: &App, stop: &AtomicBool) -> Result<()> {
    let mut report = sync::SyncReport::default();
    let mut tapo = sync::connect(&app.cfg)?;
    let started = now_ts();
    sync::discover(&app.cfg, &app.db, &app.clock, &mut tapo, &mut report)?;
    app.db.meta_set("last_discovery", &started.to_string())?;
    sync::download_pending(&app.cfg, &app.db, &app.clock, &mut tapo, stop, &mut report)?;
    log::info!(
        "camera: {} recordings listed ({} new), {} downloaded ({:.0} MB), {} failed",
        report.listed,
        report.new,
        report.downloaded,
        report.bytes as f64 / 1e6,
        report.failed
    );
    Ok(())
}

fn do_run(app: &App, stop: Arc<AtomicBool>) -> Result<()> {
    if let Err(e) = do_sync(app, &stop) {
        // Nights that are already downloaded can still be processed.
        log::error!("sync failed: {e:#}");
    }
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    pipeline::process_ready(&app.cfg, &app.db, &app.clock, stop)?;
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    if let Err(e) = real_main(cli) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn real_main(cli: Cli) -> Result<()> {
    match &cli.cmd {
        Cmd::Init { path } => {
            if path.exists() {
                bail!("{} already exists", path.display());
            }
            std::fs::write(path, config::EXAMPLE)?;
            println!("written {} – fill in [camera] and run `tapdream models fetch` and `tapdream doctor`", path.display());
        }
        Cmd::Doctor { camera } => doctor(&cli, *camera)?,
        Cmd::Camera { cmd } => {
            let app = open_app(&cli, false)?;
            let mut tapo = sync::connect(&app.cfg)?;
            match cmd {
                CameraCmd::Info => {
                    let info = tapo.device_info()?;
                    println!("transport: {}", tapo.transport_name());
                    println!("{}", serde_json::to_string_pretty(&info)?);
                    println!("clock offset: {} s", tapo.time_correction()?);
                }
                CameraCmd::List { date } => {
                    let d = match date {
                        Some(d) => parse_date(d)?,
                        None => app.clock.zone.to_local(now_ts()).date(),
                    };
                    let corr = tapo.time_correction().unwrap_or(0);
                    let recs = tapo.recordings_of_day(&d.format("%Y%m%d").to_string())?;
                    for r in &recs {
                        println!(
                            "{}  –  {}  ({:>5} s, type {})",
                            fmt_ts(&app.clock.zone, r.start + corr),
                            fmt_ts(&app.clock.zone, r.end + corr),
                            r.end - r.start,
                            r.video_type
                        );
                    }
                    println!("{} recordings", recs.len());
                }
            }
        }
        Cmd::Sync => {
            let app = open_app(&cli, true)?;
            do_sync(&app, &stop_flag())?;
        }
        Cmd::Process => {
            let app = open_app(&cli, true)?;
            pipeline::process_ready(&app.cfg, &app.db, &app.clock, stop_flag())?;
        }
        Cmd::Run => {
            let app = open_app(&cli, true)?;
            do_run(&app, stop_flag())?;
        }
        Cmd::Daemon => {
            let app = open_app(&cli, true)?;
            let stop = stop_flag();
            loop {
                if let Err(e) = do_run(&app, stop.clone()) {
                    log::error!("{e:#}");
                }
                let wait = app.cfg.schedule.daemon_interval_secs.max(60);
                log::info!("next run in {} min", wait / 60);
                for _ in 0..wait {
                    if stop.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
            }
        }
        Cmd::Status => status(&cli)?,
        Cmd::Nights => {
            let app = open_app(&cli, false)?;
            let uncensored = app.cfg.timelapse.uncensored;
            for n in app.db.list_nights()? {
                let extra = if uncensored || n.uncensored_segment.is_some() {
                    format!(" | uncensored {:>5}", n.uncensored_frames)
                } else {
                    String::new()
                };
                println!("{}  {:<9} {}{extra}", n.night, n.status, pipeline::describe_night_stats(&n.stats, app.cfg.analysis.fps));
            }
        }
        Cmd::Rebuild { reencode } => {
            let app = open_app(&cli, true)?;
            if *reencode {
                pipeline::reencode_all(&app.cfg, &app.db)?;
            } else {
                pipeline::rebuild_master(&app.cfg, &app.db)?;
            }
        }
        Cmd::Redact { from, to, also_uncensored } => {
            let app = open_app(&cli, true)?;
            let (a, b) = (pipeline::parse_local(&app.clock, from)?, pipeline::parse_local(&app.clock, to)?);
            for (v, n) in pipeline::redact(&app.cfg, &app.db, a, b, *also_uncensored)? {
                let what = if v == Variant::Uncensored { "the uncensored timelapse" } else { "the timelapse" };
                println!("removed {n} frames between {} and {} from {what}", pipeline::fmt(&app.clock, a), pipeline::fmt(&app.clock, b));
            }
            println!("this range stays excluded for future processing too");
        }
        Cmd::Reprocess { night, to } => reprocess(&cli, night, to.as_deref())?,
        Cmd::Export { output, uncensored } => {
            let app = open_app(&cli, false)?;
            let v = if *uncensored { Variant::Uncensored } else { Variant::Censored };
            let src = app.db.master(v)?.map(|m| PathBuf::from(m.path)).unwrap_or_else(|| app.cfg.output_path(v));
            if !src.exists() {
                bail!("{} does not exist yet", src.display());
            }
            if v == Variant::Uncensored {
                pipeline::night::create_private(output)?;
            }
            video::ffmpeg::export(&src, output)?;
            println!("written {}", output.display());
        }
        Cmd::Inspect { file, at } => inspect(&cli, file, *at)?,
        Cmd::Models { cmd: ModelsCmd::Fetch { no_clip } } => {
            let cfg = Config::load(&cli.config).unwrap_or_default();
            let dir = cfg.general.data_dir.join("models");
            models::fetch(&dir, !*no_clip)?;
            println!("models are in {}", dir.display());
        }
    }
    Ok(())
}

fn reprocess(cli: &Cli, first: &str, last: Option<&str>) -> Result<()> {
    let app = open_app(cli, true)?;
    let first = parse_date(first)?;
    let last = last.map(parse_date).transpose()?.unwrap_or(first);
    if last < first {
        bail!("--to is before the first day");
    }
    let stop = stop_flag();
    let mut camera: Option<tapo::Tapo> = None;
    let (mut reset, mut kept) = (0, 0);
    for day in first.iter_days().take_while(|d| *d <= last) {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let missing = pipeline::missing_recordings(&app.db, &app.clock, day)?;
        if !missing.is_empty() {
            log::info!("day {day}: downloading {} deleted recordings again", missing.len());
            if camera.is_none() {
                camera = Some(sync::connect(&app.cfg)?);
            }
            if let Err(e) = sync::redownload(&app.cfg, &app.db, &app.clock, camera.as_mut().unwrap(), &missing, &stop) {
                log::error!("day {day} stays as it is: {e:#}");
                kept += 1;
                continue;
            }
        }
        pipeline::reset_night(&app.db, &app.clock, day)?;
        reset += 1;
    }
    log::info!("{reset} days will be processed again{}", if kept > 0 { format!(", {kept} kept unchanged") } else { String::new() });
    pipeline::process_ready(&app.cfg, &app.db, &app.clock, stop)?;
    Ok(())
}

fn print_master(app: &App, v: Variant, name: &str) -> Result<()> {
    match app.db.master(v)? {
        Some(m) => {
            let size = std::fs::metadata(&m.path).map(|md| md.len()).unwrap_or(0);
            println!(
                "{name:<13}{} ({}, {:.1} MB, {} nights)",
                m.path,
                pipeline::human_duration(m.total_duration as f64 / m.timescale as f64),
                size as f64 / 1e6,
                m.segments.len()
            );
        }
        None => println!("{name:<13}not created yet"),
    }
    Ok(())
}

fn status(cli: &Cli) -> Result<()> {
    let app = open_app(cli, false)?;
    let recs = app.db.all_recordings()?;
    let count = |s: RecStatus| recs.iter().filter(|r| r.status == s).count();
    println!(
        "recordings:  {} known, {} waiting, {} downloaded, {} done & deleted, {} skipped, {} failed",
        recs.len(),
        count(RecStatus::New),
        count(RecStatus::Downloaded),
        count(RecStatus::Purged),
        count(RecStatus::Skipped),
        count(RecStatus::Failed)
    );
    let nights = app.db.list_nights()?;
    let processed: Vec<_> = nights.iter().filter(|n| n.status == "processed" || n.status == "reprocess").collect();
    let frames: i64 = processed.iter().map(|n| n.stats.emitted).sum();
    let private: i64 = processed.iter().map(|n| n.stats.skipped_private).sum();
    println!(
        "nights:      {} processed ({} with footage), {} frames in total",
        processed.len(),
        processed.iter().filter(|n| n.segment.is_some()).count(),
        frames
    );
    println!("privacy:     {:.0} min of footage withheld by the filter", private as f64 / app.cfg.analysis.fps / 60.0);
    if let (Some(first), Some(last)) = (processed.first(), processed.last()) {
        println!("covering:    {} – {}", first.night, last.night);
    }
    print_master(&app, Variant::Censored, "timelapse:")?;
    if app.cfg.timelapse.uncensored || app.db.master(Variant::Uncensored)?.is_some() {
        print_master(&app, Variant::Uncensored, "uncensored:")?;
        // nights processed before the option was switched on
        let missing: Vec<_> = processed.iter().filter(|n| n.segment.is_some() && n.uncensored_segment.is_none()).collect();
        if let (true, Some(a), Some(b)) = (app.cfg.timelapse.uncensored, missing.first(), missing.last()) {
            println!(
                "             {} older nights have no uncensored version; `tapdream reprocess {} --to {}` adds those still on the SD card",
                missing.len(),
                a.night,
                b.night
            );
        }
        if !app.cfg.timelapse.uncensored {
            println!("             (timelapse.uncensored is off: this file is no longer extended)");
        }
    }
    if let Some(e) = app.db.meta_get("encoder")? {
        println!("format:      {e}");
    }
    if let Some(t) = app.db.meta_get("last_discovery")?.and_then(|v| v.parse::<i64>().ok()) {
        println!("last sync:   {}", fmt_ts(&app.clock.zone, t));
    }
    for r in recs.iter().filter(|r| r.status == RecStatus::Failed).take(5) {
        println!("failed:      {} – {}", fmt_ts(&app.clock.zone, r.start_ts), r.last_error.as_deref().unwrap_or(""));
    }
    Ok(())
}

fn inspect(cli: &Cli, file: &std::path::Path, at: f64) -> Result<()> {
    let cfg = Config::load(&cli.config)?;
    let (w, h) = (cfg.timelapse.width, cfg.timelapse.height);
    let opts = video::ffmpeg::DecodeOptions {
        width: w,
        height: h,
        fps: cfg.analysis.fps,
        crop: cfg.timelapse.crop.clone(),
        cuda: false,
        keyframes_only: false,
        start: at,
        length: None,
        still: true,
    };
    let mut dec = video::ffmpeg::Decoder::spawn(file, &opts)?;
    let mut frame = Vec::new();
    if !dec.read_frame(&mut frame)? {
        bail!("no frame decoded from {}", file.display());
    }
    drop(dec);
    let yuv = vision::image::Yuv::new(&frame, w as usize, h as usize);
    let rt = vision::onnx::runtime(&cfg.gpu, &cfg.general.data_dir)?;
    let mut n = vision::nudenet::NudeNet::load(rt, &cfg.privacy_model(), cfg.privacy.input_size, &cfg.privacy.classes)?;
    let scores = n.class_scores(&yuv)?;
    println!("NudeNet ({}), threshold {:.2}:", n.device, cfg.privacy.threshold);
    for (i, label) in vision::nudenet::LABELS.iter().enumerate() {
        let watched = cfg.privacy.classes.iter().any(|c| c.eq_ignore_ascii_case(label));
        if scores[i] >= 0.05 || watched {
            println!("  {:<26} {:.3}{}", label, scores[i], if watched && scores[i] >= cfg.privacy.threshold { "  <- PRIVATE" } else { "" });
        }
    }
    if cfg.scene.enabled {
        let s = &cfg.scene;
        let g = vision::clip::PromptGroups {
            presence: &s.presence_prompts,
            empty: &s.empty_prompts,
            interest: &s.interest_prompts,
            neutral: &s.neutral_prompts,
            private: &cfg.privacy.clip_prompts,
            contrast: &cfg.privacy.clip_contrast_prompts,
        };
        let mut c = vision::clip::Clip::load(rt, &cfg.model_path(&s.visual_model), &cfg.model_path(&s.textual_model), &g)?;
        let sc = c.scores(&yuv)?;
        let emb = c.embed(&yuv)?;
        log::trace!("clip embedding {:?}", emb);
        println!("CLIP ({}):", c.device);
        println!("  presence {:.3} (threshold {:.2})", sc.presence, s.presence_threshold);
        println!("  interest {:.3}", sc.interest);
        println!("  private  {:.3} (threshold {:.2})", sc.private, cfg.privacy.clip_threshold);
    }
    Ok(())
}

fn doctor(cli: &Cli, camera: bool) -> Result<()> {
    let cfg = Config::load(&cli.config)?;
    let ok = |b: bool| if b { "ok " } else { "-- " };
    let caps = video::ffmpeg::caps();
    match &caps {
        Ok(c) => {
            println!("[ok ] {}", c.version);
            println!("[{}] NVIDIA GPU device", ok(c.nvidia));
            println!("[{}] NVDEC decoding (ffmpeg -hwaccel cuda)", ok(c.hwaccels.contains("cuda")));
            for e in ["hevc_nvenc", "h264_nvenc", "libx265", "libx264"] {
                println!("[{}] encoder {e}", ok(c.encoders.contains(e)));
            }
            match video::ffmpeg::EncoderSpec::choose(
                c,
                &cfg.timelapse.codec,
                &cfg.timelapse.encoder,
                cfg.timelapse.quality,
                cfg.timelapse.width,
                cfg.timelapse.height,
                cfg.timelapse.fps,
            ) {
                Ok(s) => println!("[ok ] timelapse encoder: {}", s.name),
                Err(e) => println!("[!! ] {e:#}"),
            }
        }
        Err(e) => println!("[!! ] {e:#}"),
    }
    match vision::onnx::runtime(&cfg.gpu, &cfg.general.data_dir) {
        Ok(rt) => {
            println!("[ok ] ONNX Runtime {} (provider: {})", rt.lib.display(), rt.provider);
            match vision::nudenet::NudeNet::load(rt, &cfg.privacy_model(), cfg.privacy.input_size, &cfg.privacy.classes) {
                Ok(n) => println!("[ok ] nudity filter {} on {}", cfg.privacy_model().display(), n.device),
                Err(e) => println!("[!! ] nudity filter: {e:#}"),
            }
            if cfg.scene.enabled {
                let s = &cfg.scene;
                let g = vision::clip::PromptGroups {
                    presence: &s.presence_prompts,
                    empty: &s.empty_prompts,
                    interest: &s.interest_prompts,
                    neutral: &s.neutral_prompts,
                    private: &cfg.privacy.clip_prompts,
                    contrast: &cfg.privacy.clip_contrast_prompts,
                };
                match vision::clip::Clip::load(rt, &cfg.model_path(&s.visual_model), &cfg.model_path(&s.textual_model), &g) {
                    Ok(c) => println!("[ok ] scene model (CLIP) on {}", c.device),
                    Err(e) => println!("[-- ] scene model: {e:#}"),
                }
            }
        }
        Err(e) => println!("[!! ] {e:#}"),
    }
    match video::overlay::find_font(&cfg.overlay.font) {
        Some(f) => println!("[ok ] caption font {}", f.display()),
        None => println!("[-- ] no caption font found"),
    }
    if camera {
        match sync::connect(&cfg) {
            Ok(mut t) => {
                println!("[ok ] camera {} ({})", cfg.camera.host, t.transport_name());
                match t.user_id() {
                    Ok(_) => println!("[ok ] playback access"),
                    Err(e) => println!("[!! ] playback access: {e:#}"),
                }
            }
            Err(e) => println!("[!! ] camera: {e:#}"),
        }
    }
    Ok(())
}
