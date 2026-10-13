//! Configuration (TOML). Every field has a sensible default except the camera
//! address and the TP-Link cloud password.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub general: General,
    pub camera: Camera,
    pub schedule: Schedule,
    pub timelapse: Timelapse,
    pub analysis: Analysis,
    pub privacy: Privacy,
    pub scene: Scene,
    pub overlay: Overlay,
    pub gpu: Gpu,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct General {
    /// Directory holding the index, downloaded recordings, segments and the timelapse.
    pub data_dir: PathBuf,
    /// IANA time zone used to split recordings into nights. Empty = system local time.
    pub timezone: String,
}

impl Default for General {
    fn default() -> Self {
        Self { data_dir: PathBuf::from("./tapdream-data"), timezone: String::new() }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Camera {
    pub host: String,
    /// Password of the TP-Link / Tapo cloud account the camera is bound to.
    /// Can also be given via TAPDREAM_CLOUD_PASSWORD or `cloud_password_file`.
    pub cloud_password: String,
    pub cloud_password_file: String,
    /// auto | classic | tpap
    pub transport: String,
    pub control_port: u16,
    pub media_port: u16,
    /// download (fast, ~10x realtime) | playback (realtime, for old firmware)
    pub download_method: String,
    pub window_size: u32,
    pub stall_timeout_secs: u64,
    /// How many days back to look for recordings.
    pub history_days: u32,
    /// Ignore anything that started before this date (YYYY-MM-DD). Empty = no limit.
    pub since: String,
    /// Recordings that ended less than this many seconds ago may still be in progress.
    pub min_age_secs: i64,
    pub max_attempts: u32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            host: String::new(),
            cloud_password: String::new(),
            cloud_password_file: String::new(),
            transport: "auto".into(),
            control_port: 443,
            media_port: 8800,
            download_method: "download".into(),
            window_size: 200,
            stall_timeout_secs: 120,
            history_days: 30,
            since: String::new(),
            min_age_secs: 120,
            max_attempts: 4,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Schedule {
    /// A "night" lasts from this local time until the same time the next day.
    pub day_boundary: String,
    /// Only footage inside this local time window is used, e.g. "18:00-10:00". Empty = all day.
    pub capture_hours: String,
    /// Minutes to wait after a night ended before it is processed.
    pub grace_minutes: i64,
    /// Keep downloaded recordings after their night was processed.
    pub keep_downloads: bool,
    /// Seconds between runs in `daemon` mode.
    pub daemon_interval_secs: u64,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            day_boundary: "12:00".into(),
            capture_hours: String::new(),
            grace_minutes: 30,
            keep_downloads: false,
            daemon_interval_secs: 3600,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Timelapse {
    /// The ever-growing timelapse file (relative paths are inside data_dir).
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// Optional crop of the source picture before scaling, ffmpeg syntax "w:h:x:y".
    pub crop: String,
    /// hevc | h264
    pub codec: String,
    /// auto | nvenc | software
    pub encoder: String,
    /// Constant quality (NVENC -cq / x264,x265 -crf). Lower = better.
    pub quality: u32,
    /// One output frame per this many seconds of real time while something happens.
    pub interval_secs: f64,
    /// ... while the scene is calm (someone is there but barely moves, e.g. sleep).
    pub calm_interval_secs: f64,
    /// ... while the scene is empty/static (camera left running). 0 = drop entirely.
    pub idle_interval_secs: f64,
    /// Save the most interesting frame of every night as JPEG into data_dir/highlights.
    pub highlights: bool,
}

impl Default for Timelapse {
    fn default() -> Self {
        Self {
            output: PathBuf::from("timelapse.mp4"),
            width: 1920,
            height: 1080,
            fps: 30,
            crop: String::new(),
            codec: "hevc".into(),
            encoder: "auto".into(),
            quality: 24,
            interval_secs: 20.0,
            calm_interval_secs: 90.0,
            idle_interval_secs: 900.0,
            highlights: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Analysis {
    /// Frames per second sampled from the recordings for analysis.
    pub fps: f64,
    /// auto | all | keyframes (keyframes-only decoding is many times faster)
    pub decode: String,
    /// auto | cuda | none
    pub hwaccel: String,
    /// Per-pixel luma difference (0-255) on the downscaled frame that counts as change.
    pub motion_pixel_threshold: u8,
    /// Fraction of changed pixels that counts as activity.
    pub motion_threshold: f64,
    /// Activity keeps the timelapse at full speed this long after the last motion.
    pub activity_hold_secs: f64,
    /// Without the scene model: a scene static for this long is considered idle.
    pub idle_after_secs: f64,
    /// Mean luma below this = black frame, never used.
    pub dark_threshold: f64,
}

impl Default for Analysis {
    fn default() -> Self {
        Self {
            fps: 1.0,
            decode: "auto".into(),
            hwaccel: "auto".into(),
            motion_pixel_threshold: 14,
            motion_threshold: 0.004,
            activity_hold_secs: 30.0,
            idle_after_secs: 900.0,
            dark_threshold: 10.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Privacy {
    /// When true (default) processing refuses to run without a working nudity detector.
    pub required: bool,
    /// NudeNet ONNX model (320n or 640m).
    pub model: PathBuf,
    /// Model input resolution, 0 = derive from the file name (320/640).
    pub input_size: u32,
    pub threshold: f32,
    pub classes: Vec<String>,
    /// Frames this many seconds before a detection are dropped too.
    pub before_secs: f64,
    /// ... and this many seconds after.
    pub after_secs: f64,
    /// Reuse the previous verdict while the picture changed less than this (fraction of pixels).
    pub reuse_below_change: f64,
    /// Extra CLIP prompts treated as private (needs the scene model).
    pub clip_prompts: Vec<String>,
    /// Ordinary situations the private prompts are contrasted with (besides the scene prompts).
    pub clip_contrast_prompts: Vec<String>,
    /// Probability (vs. the ordinary scene prompts) above which a CLIP private prompt counts as a hit.
    pub clip_threshold: f32,
    /// CLIP alone triggers only after this many hits within 90 s (it is a
    /// second opinion next to NudeNet; single hits are often false alarms).
    pub clip_min_hits: u32,
}

impl Default for Privacy {
    fn default() -> Self {
        Self {
            required: true,
            model: PathBuf::from("models/nudenet-640m.onnx"),
            input_size: 0,
            threshold: 0.25,
            classes: vec![
                "FEMALE_BREAST_EXPOSED".into(),
                "FEMALE_GENITALIA_EXPOSED".into(),
                "MALE_GENITALIA_EXPOSED".into(),
                "BUTTOCKS_EXPOSED".into(),
                "ANUS_EXPOSED".into(),
            ],
            before_secs: 120.0,
            after_secs: 300.0,
            reuse_below_change: 0.0015,
            clip_prompts: vec![
                "a photo of a woman breastfeeding a baby".into(),
                "a photo of a mother nursing her baby at her breast".into(),
            ],
            clip_contrast_prompts: vec![
                "a photo of a woman holding a baby".into(),
                "a photo of a man holding a baby".into(),
                "a photo of a woman".into(),
                "a photo of a man".into(),
                "a photo of a person".into(),
            ],
            clip_threshold: 0.6,
            clip_min_hits: 2,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Scene {
    pub enabled: bool,
    pub visual_model: PathBuf,
    pub textual_model: PathBuf,
    /// Run CLIP at most once per this many seconds of footage (plus on candidates).
    pub every_secs: f64,
    /// Somebody (baby/parents) is in the picture.
    pub presence_prompts: Vec<String>,
    /// Nobody is there.
    pub empty_prompts: Vec<String>,
    /// Moments worth keeping.
    pub interest_prompts: Vec<String>,
    /// Ordinary moments.
    pub neutral_prompts: Vec<String>,
    /// Presence probability below this = empty scene (idle).
    pub presence_threshold: f32,
    /// How much interest pulls the playback speed down (0 = off, 1 = interesting moments at full speed).
    pub interest_weight: f32,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            enabled: true,
            visual_model: PathBuf::from("models/clip-vit-b32-visual.onnx"),
            textual_model: PathBuf::from("models/clip-vit-b32-textual.onnx"),
            every_secs: 10.0,
            presence_prompts: vec![
                "a photo of a baby sleeping in a crib".into(),
                "a photo of a baby lying in a bed".into(),
                "a photo of a parent holding a baby".into(),
                "a photo of people lying in a bed".into(),
            ],
            empty_prompts: vec![
                "a photo of an empty crib".into(),
                "a photo of an empty bed".into(),
                "a photo of an empty dark bedroom".into(),
            ],
            interest_prompts: vec![
                "a photo of a smiling baby".into(),
                "a photo of a baby awake and looking around".into(),
                "a photo of a family cuddling together".into(),
                "a photo of a baby stretching its arms".into(),
                "a photo of a parent kissing a baby".into(),
            ],
            neutral_prompts: vec![
                "a photo of a baby sleeping".into(),
                "a photo of a bedroom at night".into(),
                "a blurry dark photo".into(),
            ],
            presence_threshold: 0.35,
            interest_weight: 0.7,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Overlay {
    pub enabled: bool,
    /// TTF/OTF font. Empty = first usable system font.
    pub font: String,
    /// Placeholders: {night} {date} {age} {time} {weekday}
    pub template: String,
    /// Night number 1. Empty = birth_date, or the first processed night.
    pub start_date: String,
    /// Baby's birth date (YYYY-MM-DD) for {age}.
    pub birth_date: String,
    /// cs | en
    pub language: String,
    /// bottom-left | bottom-right | top-left | top-right
    pub position: String,
    /// Text height as a fraction of the frame height.
    pub size: f32,
}

impl Default for Overlay {
    fn default() -> Self {
        Self {
            enabled: true,
            font: String::new(),
            template: "Noc {night} · {date}".into(),
            start_date: String::new(),
            birth_date: String::new(),
            language: "cs".into(),
            position: "bottom-left".into(),
            size: 0.035,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Gpu {
    /// libonnxruntime.so to load. Empty = $ORT_DYLIB_PATH, then common install locations.
    pub onnxruntime_lib: String,
    /// auto | cuda | tensorrt | cpu
    pub provider: String,
    pub device_id: i32,
    /// CPU threads for ONNX Runtime when running on CPU (0 = ONNX Runtime default).
    pub cpu_threads: usize,
}

impl Default for Gpu {
    fn default() -> Self {
        Self { onnxruntime_lib: String::new(), provider: "auto".into(), device_id: 0, cpu_threads: 0 }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("cannot read config {}", path.display()))?;
        let mut cfg: Config = toml::from_str(&text).with_context(|| format!("invalid config {}", path.display()))?;
        // Relative paths in the config are relative to the config file, not the CWD.
        let base = path.parent().map(Path::to_path_buf).unwrap_or_default();
        if cfg.general.data_dir.is_relative() {
            cfg.general.data_dir = base.join(&cfg.general.data_dir);
        }
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if !self.timelapse.width.is_multiple_of(2) || !self.timelapse.height.is_multiple_of(2) {
            bail!("timelapse.width/height must be even");
        }
        if self.timelapse.fps == 0 || self.timelapse.fps > 120 {
            bail!("timelapse.fps must be 1..=120");
        }
        if !(self.analysis.fps > 0.0 && self.analysis.fps <= 10.0) {
            bail!("analysis.fps must be in (0, 10]");
        }
        if self.timelapse.interval_secs <= 0.0 || self.timelapse.calm_interval_secs <= 0.0 {
            bail!("timelapse intervals must be positive");
        }
        if self.timelapse.idle_interval_secs < 0.0 {
            bail!("timelapse.idle_interval_secs must be >= 0");
        }
        match self.timelapse.codec.as_str() {
            "hevc" | "h264" => {}
            other => bail!("timelapse.codec must be hevc or h264, not {other:?}"),
        }
        Ok(())
    }

    pub fn data_path(&self, p: &Path) -> PathBuf {
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.general.data_dir.join(p)
        }
    }

    pub fn output_path(&self) -> PathBuf {
        self.data_path(&self.timelapse.output)
    }

    /// Model paths are resolved against data_dir.
    pub fn model_path(&self, p: &Path) -> PathBuf {
        self.data_path(p)
    }

    /// The NudeNet model; falls back to the small 320n model next to it when
    /// the configured one is missing (e.g. the 640m download failed).
    pub fn privacy_model(&self) -> PathBuf {
        let p = self.model_path(&self.privacy.model);
        if !p.exists() {
            let small = p.with_file_name("nudenet-320n.onnx");
            if small.exists() {
                log::warn!("{} not found, using {}", p.display(), small.display());
                return small;
            }
        }
        p
    }

    pub fn cloud_password(&self) -> Result<String> {
        if let Ok(p) = std::env::var("TAPDREAM_CLOUD_PASSWORD") {
            if !p.is_empty() {
                return Ok(p);
            }
        }
        if !self.camera.cloud_password_file.is_empty() {
            let p = std::fs::read_to_string(&self.camera.cloud_password_file)
                .with_context(|| format!("cannot read {}", self.camera.cloud_password_file))?;
            return Ok(p.trim_end_matches(['\r', '\n']).to_string());
        }
        if !self.camera.cloud_password.is_empty() {
            return Ok(self.camera.cloud_password.clone());
        }
        bail!("camera cloud password is not set (camera.cloud_password, camera.cloud_password_file or TAPDREAM_CLOUD_PASSWORD)")
    }
}

pub const EXAMPLE: &str = include_str!("../config.example.toml");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_config_parses() {
        let cfg: Config = toml::from_str(EXAMPLE).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.camera.media_port, 8800);
    }
}
