//! Frame selection – the "intelligent" part of the timelapse.
//!
//! * Adaptive speed: a virtual clock decides how much real time one output
//!   frame represents. Activity → `interval`, calm (someone is there, e.g.
//!   sleeping) → `calm_interval` (pulled towards `interval` by interesting
//!   moments), empty/static scene → `idle_interval` or nothing at all.
//! * Within each output slot the best candidate wins (sharp, interesting).
//! * Privacy: a detection blocks a window *around* it. Selected frames wait
//!   in a delay queue for `before` seconds, so frames that preceded a
//!   detection are dropped too – nothing private ever reaches the encoder.

use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct SelectorConfig {
    pub analysis_fps: f64,
    pub interval: f64,
    pub calm_interval: f64,
    /// 0 = idle footage is dropped completely
    pub idle_interval: f64,
    pub motion_threshold: f64,
    pub activity_hold: f64,
    pub idle_after: f64,
    pub presence_threshold: f32,
    pub interest_weight: f32,
    pub before: f64,
    pub after: f64,
}

#[derive(Clone, Debug, Default)]
pub struct FrameInfo {
    /// Unix time of the frame (fractional).
    pub t: f64,
    /// Fraction of the picture that changed since the previous analysed frame.
    pub motion: f32,
    pub sharpness: f32,
    /// None when the scene model is disabled.
    pub presence: Option<f32>,
    pub interest: f32,
    /// Normalized private score (>= 1.0 means a detection).
    pub private: f32,
    pub dark: bool,
}

impl FrameInfo {
    pub fn score(&self) -> f32 {
        self.sharpness.max(0.0).sqrt() * (1.0 + 2.0 * self.interest)
    }
}

#[derive(Debug, Default, Clone)]
pub struct SelectorStats {
    pub analyzed: u64,
    pub dark: u64,
    pub idle: u64,
    pub private: u64,
    pub emitted: u64,
    pub dropped_private: u64,
}

pub struct Selector<P> {
    cfg: SelectorConfig,
    budget: f64,
    last_t: Option<f64>,
    last_active: f64,
    last_change: f64,
    best: Option<(FrameInfo, P)>,
    pending: VecDeque<(FrameInfo, P)>,
    blocked: Vec<(f64, f64)>,
    pub stats: SelectorStats,
}

impl<P> Selector<P> {
    pub fn new(cfg: SelectorConfig, manual_blocks: Vec<(f64, f64)>) -> Self {
        Self {
            cfg,
            budget: 0.0,
            last_t: None,
            last_active: f64::NEG_INFINITY,
            last_change: f64::NEG_INFINITY,
            best: None,
            pending: VecDeque::new(),
            blocked: manual_blocks,
            stats: SelectorStats::default(),
        }
    }

    fn is_blocked(&self, t: f64) -> bool {
        self.blocked.iter().any(|&(a, b)| t >= a && t <= b)
    }

    /// Feeds one analysed frame. `payload` is only called when the frame
    /// becomes the current best candidate (avoids copying every frame).
    /// Returns frames that are final and can be encoded.
    pub fn push(&mut self, info: FrameInfo, payload: impl FnOnce() -> P) -> Vec<(FrameInfo, P)> {
        let c = &self.cfg;
        let t = info.t;
        self.stats.analyzed += 1;
        let step = 1.0 / c.analysis_fps;
        // gaps between recordings must not turn into a burst of frames
        let dt = self.last_t.map(|l| (t - l).clamp(0.0, 2.0 * step)).unwrap_or(step);
        self.last_t = Some(t);
        if self.last_change == f64::NEG_INFINITY {
            self.last_change = t;
        }

        if info.private >= 1.0 {
            self.stats.private += 1;
            self.blocked.push((t - c.before, t + c.after));
            if self.best.as_ref().map(|(b, _)| b.t >= t - c.before).unwrap_or(false) {
                self.best = None;
            }
            return self.release(t);
        }
        if info.dark {
            self.stats.dark += 1;
            return self.release(t);
        }
        if info.motion as f64 >= c.motion_threshold {
            self.last_active = t;
            self.last_change = t;
        }
        let active = t - self.last_active <= c.activity_hold;
        let idle = !active
            && match info.presence {
                Some(p) => p < c.presence_threshold,
                None => t - self.last_change > c.idle_after,
            };
        let interval = if idle {
            self.stats.idle += 1;
            c.idle_interval
        } else if active {
            c.interval
        } else {
            let w = (c.interest_weight * info.interest).clamp(0.0, 1.0) as f64;
            c.calm_interval + (c.interval - c.calm_interval) * w
        };
        if interval <= 0.0 {
            // idle footage is dropped entirely; start a fresh slot afterwards
            self.best = None;
            self.budget = 0.0;
            return self.release(t);
        }
        if !self.is_blocked(t) {
            let better = self.best.as_ref().map(|(b, _)| info.score() > b.score()).unwrap_or(true);
            if better {
                let p = payload();
                self.best = Some((info, p));
            }
        }
        self.budget += dt / interval;
        if self.budget >= 1.0 {
            self.budget = (self.budget - 1.0).min(1.0);
            if let Some(b) = self.best.take() {
                self.pending.push_back(b);
            }
        }
        self.release(t)
    }

    /// Frames older than the look-back window can no longer be affected by a
    /// future detection.
    fn release(&mut self, now: f64) -> Vec<(FrameInfo, P)> {
        let horizon = now - self.cfg.before;
        let mut out = Vec::new();
        while self.pending.front().map(|(f, _)| f.t < horizon).unwrap_or(false) {
            let (f, p) = self.pending.pop_front().unwrap();
            if self.is_blocked(f.t) {
                self.stats.dropped_private += 1;
            } else {
                self.stats.emitted += 1;
                out.push((f, p));
            }
        }
        out
    }

    /// End of the night: flush everything that is not blocked.
    pub fn finish(&mut self) -> Vec<(FrameInfo, P)> {
        if let Some(b) = self.best.take() {
            if self.budget >= 0.5 {
                self.pending.push_back(b);
            }
        }
        self.release(f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SelectorConfig {
        SelectorConfig {
            analysis_fps: 1.0,
            interval: 10.0,
            calm_interval: 60.0,
            idle_interval: 0.0,
            motion_threshold: 0.01,
            activity_hold: 5.0,
            idle_after: 120.0,
            presence_threshold: 0.4,
            interest_weight: 0.0,
            before: 30.0,
            after: 60.0,
        }
    }

    fn frame(t: f64, motion: f32) -> FrameInfo {
        FrameInfo { t, motion, sharpness: 100.0, presence: Some(0.9), ..Default::default() }
    }

    fn run(sel: &mut Selector<f64>, frames: Vec<FrameInfo>) -> Vec<f64> {
        let mut out = Vec::new();
        for f in frames {
            let t = f.t;
            out.extend(sel.push(f, || t).into_iter().map(|(_, p)| p));
        }
        out.extend(sel.finish().into_iter().map(|(_, p)| p));
        out
    }

    #[test]
    fn adaptive_speed() {
        // 10 minutes of activity then 10 minutes of calm presence
        let mut s = Selector::new(cfg(), vec![]);
        let frames: Vec<_> = (0..1200).map(|i| frame(i as f64, if i < 600 { 0.05 } else { 0.0 })).collect();
        let out = run(&mut s, frames);
        let active = out.iter().filter(|&&t| t < 600.0).count();
        let calm = out.iter().filter(|&&t| t >= 600.0).count();
        assert!((59..=61).contains(&active), "{active}");
        assert!((10..=12).contains(&calm), "{calm}");
    }

    #[test]
    fn empty_scene_is_dropped() {
        let mut s = Selector::new(cfg(), vec![]);
        let frames: Vec<_> = (0..600).map(|i| FrameInfo { presence: Some(0.05), ..frame(i as f64, 0.0) }).collect();
        assert!(run(&mut s, frames).is_empty());
        assert_eq!(s.stats.idle, 600);
    }

    #[test]
    fn privacy_window_reaches_back() {
        let mut s = Selector::new(cfg(), vec![]);
        let mut frames: Vec<_> = (0..600).map(|i| frame(i as f64, 0.05)).collect();
        frames[300].private = 1.5;
        let out = run(&mut s, frames);
        // nothing in [270, 360]
        assert!(out.iter().all(|&t| !(270.0..=360.0).contains(&t)), "{out:?}");
        assert!(out.iter().any(|&t| t < 260.0) && out.iter().any(|&t| t > 370.0));
        assert!(s.stats.dropped_private > 0 || s.stats.private > 0);
    }

    #[test]
    fn manual_redaction_blocks() {
        let mut s = Selector::new(cfg(), vec![(100.0, 200.0)]);
        let out = run(&mut s, (0..400).map(|i| frame(i as f64, 0.05)).collect());
        assert!(out.iter().all(|&t| !(100.0..=200.0).contains(&t)));
    }

    #[test]
    fn best_candidate_wins() {
        let mut s = Selector::new(cfg(), vec![]);
        let frames: Vec<_> = (0..10).map(|i| FrameInfo { sharpness: if i == 7 { 900.0 } else { 10.0 }, ..frame(i as f64, 0.05) }).collect();
        assert_eq!(run(&mut s, frames), vec![7.0]);
    }
}
