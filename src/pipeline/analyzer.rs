//! Per-frame analysis: motion, sharpness, darkness, privacy and scene scores.

use super::select::FrameInfo;
use crate::config::Config;
use crate::vision::clip::{Clip, PromptGroups, SceneScores};
use crate::vision::image::{change_fraction, luma_thumb, mean, sharpness, Yuv};
use crate::vision::nudenet::NudeNet;
use crate::vision::onnx;
use anyhow::{bail, Result};

const THUMB_W: usize = 160;
const THUMB_H: usize = 90;

pub struct Models {
    pub nudenet: Option<NudeNet>,
    pub clip: Option<Clip>,
}

impl Models {
    pub fn load(cfg: &Config) -> Result<Self> {
        let rt = match onnx::runtime(&cfg.gpu, &cfg.general.data_dir) {
            Ok(rt) => Some(rt),
            Err(e) => {
                if cfg.privacy.required {
                    bail!("{e:#}\nThe nudity filter needs ONNX Runtime. Refusing to continue (privacy.required = true).");
                }
                log::warn!("{e:#}; continuing WITHOUT any AI analysis");
                None
            }
        };
        let nudenet = match rt {
            Some(rt) => match NudeNet::load(rt, &cfg.privacy_model(), cfg.privacy.input_size, &cfg.privacy.classes) {
                Ok(n) => {
                    log::info!("nudity filter: {} on {}", cfg.privacy_model().display(), n.device);
                    Some(n)
                }
                Err(e) if cfg.privacy.required => {
                    bail!("cannot load the nudity filter: {e:#}\nRun `tapdream models fetch`. Refusing to continue (privacy.required = true).")
                }
                Err(e) => {
                    log::warn!("nudity filter disabled: {e:#}");
                    None
                }
            },
            None => None,
        };
        let clip = match (rt, cfg.scene.enabled) {
            (Some(rt), true) => {
                let s = &cfg.scene;
                let groups = PromptGroups {
                    presence: &s.presence_prompts,
                    empty: &s.empty_prompts,
                    interest: &s.interest_prompts,
                    neutral: &s.neutral_prompts,
                    private: &cfg.privacy.clip_prompts,
                    contrast: &cfg.privacy.clip_contrast_prompts,
                };
                match Clip::load(rt, &cfg.model_path(&s.visual_model), &cfg.model_path(&s.textual_model), &groups) {
                    Ok(c) => {
                        log::info!("scene model: CLIP on {}", c.device);
                        Some(c)
                    }
                    Err(e) => {
                        log::warn!("scene model disabled: {e:#}");
                        None
                    }
                }
            }
            _ => None,
        };
        Ok(Self { nudenet, clip })
    }
}

/// Stateful analyser (keeps the previous frame for motion and result reuse).
pub struct Analyzer<'m> {
    models: &'m mut Models,
    cfg: Config,
    prev_thumb: Option<Vec<u8>>,
    /// thumbnail and time of the last frame the detectors actually ran on
    nsfw_ref: Option<(Vec<u8>, f64, f32)>,
    clip_ref: Option<(Vec<u8>, f64, SceneScores)>,
    clip_hits: std::collections::VecDeque<f64>,
    pub nsfw_runs: u64,
    pub clip_runs: u64,
    pub last_label: &'static str,
}

impl<'m> Analyzer<'m> {
    pub fn new(models: &'m mut Models, cfg: &Config) -> Self {
        Self {
            models,
            cfg: cfg.clone(),
            prev_thumb: None,
            nsfw_ref: None,
            clip_ref: None,
            clip_hits: Default::default(),
            nsfw_runs: 0,
            clip_runs: 0,
            last_label: "",
        }
    }

    /// Forget the previous frame (start of a non-contiguous recording).
    pub fn reset_motion(&mut self) {
        self.prev_thumb = None;
    }

    pub fn analyze(&mut self, t: f64, frame: &Yuv) -> Result<FrameInfo> {
        let a = &self.cfg.analysis;
        let thumb = luma_thumb(frame, THUMB_W, THUMB_H);
        let brightness = mean(&thumb);
        let motion = match &self.prev_thumb {
            Some(p) => change_fraction(p, &thumb, a.motion_pixel_threshold) as f32,
            None => 0.0,
        };
        let sharp = sharpness(&thumb, THUMB_W, THUMB_H) as f32;
        self.prev_thumb = Some(thumb.clone());
        let mut info = FrameInfo { t, motion, sharpness: sharp, dark: brightness < a.dark_threshold, ..Default::default() };
        if info.dark {
            return Ok(info);
        }

        // Nudity detector: rerun unless the picture is practically unchanged
        // since the last run (and that run was recent).
        let p = &self.cfg.privacy;
        if let Some(n) = self.models.nudenet.as_mut() {
            let reuse = self.nsfw_ref.as_ref().and_then(|(rt, rtime, score)| {
                (t - rtime < 60.0 && change_fraction(rt, &thumb, a.motion_pixel_threshold) < p.reuse_below_change).then_some(*score)
            });
            let score = match reuse {
                Some(s) => s,
                None => {
                    let (s, label) = n.private_score(frame)?;
                    self.nsfw_runs += 1;
                    self.last_label = label;
                    self.nsfw_ref = Some((thumb.clone(), t, s));
                    s
                }
            };
            info.private = score / p.threshold.max(1e-3);
        }

        if let Some(c) = self.models.clip.as_mut() {
            let s = &self.cfg.scene;
            let stale = match &self.clip_ref {
                None => true,
                Some((rt, rtime, _)) => t - rtime >= s.every_secs || change_fraction(rt, &thumb, a.motion_pixel_threshold) > 0.05,
            };
            if stale {
                let scores = c.scores(frame)?;
                self.clip_runs += 1;
                if !p.clip_prompts.is_empty() && scores.private >= p.clip_threshold {
                    self.clip_hits.push_back(t);
                }
                self.clip_ref = Some((thumb, t, scores));
            }
            while self.clip_hits.front().map(|&h| t - h > 90.0).unwrap_or(false) {
                self.clip_hits.pop_front();
            }
            let sc = self.clip_ref.as_ref().unwrap().2;
            info.presence = Some(sc.presence);
            info.interest = sc.interest;
            if self.clip_hits.len() >= p.clip_min_hits.max(1) as usize {
                info.private = info.private.max(sc.private / p.clip_threshold.max(1e-3)).max(1.0);
            }
        }
        Ok(info)
    }
}
