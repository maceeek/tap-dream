//! Frame selection – the "intelligent" part of the timelapse.
//!
//! * Adaptive speed: a virtual clock decides how much real time one output
//!   frame represents. Activity → `interval`, calm (someone is there, e.g.
//!   sleeping) → `calm_interval` (pulled towards `interval` by interesting
//!   moments), empty/static scene → `idle_interval` or nothing at all.
//! * Within each output slot the best candidate wins (sharp, interesting).
//! * Naps: daytime footage only counts while somebody is in the picture, and
//!   only once that has lasted `nap_min` – until then its frames are held
//!   back and dropped if it turns out to be a short visit.
//! * Privacy: a detection blocks a window *around* it. Selected frames wait
//!   in a delay queue for `before` seconds, so frames that preceded a
//!   detection are dropped too – nothing private ever reaches the encoder.
//!   Context frames (just outside the used footage) are only checked for
//!   privacy, so the window works across the edges as well.

pub use crate::timeutil::Part;
use std::collections::VecDeque;

/// Without any footage for this long a nap is considered over.
const MAX_FOOTAGE_GAP: f64 = 1800.0;

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
    /// Minimum nap length (seconds).
    pub nap_min: f64,
    /// A nap ends after this long without anybody in the picture (seconds).
    pub nap_gap: f64,
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
    pub part: Part,
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
    pub emitted_nap: u64,
    pub dropped_private: u64,
    /// Frames of daytime visits too short to be a nap.
    pub dropped_short: u64,
}

struct Nap {
    start: f64,
    last_seen: f64,
    last_frame: f64,
    qualified: bool,
    /// The sleep runs over the edge of the night window: it is part of the night.
    night_like: bool,
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
    nap: Option<Nap>,
    /// Frames of a nap that has not lasted long enough yet.
    gate: Vec<(FrameInfo, P)>,
    /// Last night frame with somebody in the picture.
    last_night_seen: f64,
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
            nap: None,
            gate: Vec::new(),
            last_night_seen: f64::NEG_INFINITY,
            stats: SelectorStats::default(),
        }
    }

    fn end_nap(&mut self) {
        if let Some(n) = self.nap.take() {
            log::debug!(
                "daytime presence {:.0}-{:.0} ({:.1} min): {}",
                n.start,
                n.last_seen,
                (n.last_seen - n.start) / 60.0,
                if n.qualified { "nap".to_string() } else { format!("too short, {} frames dropped", self.gate.len()) }
            );
            if n.night_like {
                if let Some(b) = self.best.as_mut().filter(|b| b.0.part == Part::Nap) {
                    b.0.part = Part::Night;
                }
            }
            if !n.qualified {
                self.stats.dropped_short += self.gate.len() as u64;
                self.gate.clear();
                if self.best.as_ref().map(|(b, _)| b.part == Part::Nap && b.t >= n.start).unwrap_or(false) {
                    self.best = None;
                }
            }
        }
    }

    /// Starts, extends or ends the current nap.
    fn track_nap(&mut self, t: f64, occupied: bool) {
        let (nap_min, nap_gap) = (self.cfg.nap_min, self.cfg.nap_gap);
        if self.nap.as_ref().map(|n| t - n.last_frame > MAX_FOOTAGE_GAP).unwrap_or(false) {
            self.end_nap();
        }
        if occupied {
            // sleep that simply goes on after the night window is no "short visit"
            let continues_night = t - self.last_night_seen <= nap_gap;
            let n = self.nap.get_or_insert(Nap {
                start: t,
                last_seen: t,
                last_frame: t,
                qualified: continues_night,
                night_like: continues_night,
            });
            n.last_seen = t;
            if !n.qualified && n.last_seen - n.start >= nap_min {
                n.qualified = true;
                self.pending.extend(self.gate.drain(..));
            }
        } else if self.nap.as_ref().map(|n| t - n.last_seen >= nap_gap).unwrap_or(false) {
            self.end_nap();
        }
        if let Some(n) = self.nap.as_mut() {
            n.last_frame = t;
        }
    }

    fn is_blocked(&self, t: f64) -> bool {
        self.blocked.iter().any(|&(a, b)| t >= a && t <= b)
    }

    /// Feeds one analysed frame. `payload` is only called when the frame
    /// becomes the current best candidate (avoids copying every frame).
    /// Returns frames that are final and can be encoded.
    pub fn push(&mut self, info: FrameInfo, payload: impl FnOnce() -> P) -> Vec<(FrameInfo, P)> {
        let t = info.t;
        if info.part == Part::Context {
            if info.private >= 1.0 {
                self.block(t);
            }
            return self.release(t);
        }
        if info.part != Part::Nap {
            // somebody fell asleep shortly before the night window: same sleep, keep it
            let leads_into_night = self.nap.as_ref().map(|n| !n.qualified && t - n.last_seen <= self.cfg.nap_gap).unwrap_or(false);
            if leads_into_night {
                for g in self.gate.iter_mut().chain(self.best.iter_mut()) {
                    if g.0.part == Part::Nap {
                        g.0.part = Part::Night;
                    }
                }
                self.pending.extend(self.gate.drain(..));
                if let Some(n) = self.nap.as_mut() {
                    n.qualified = true;
                }
            }
            self.end_nap();
        }
        let c = &self.cfg;
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
            self.block(t);
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
        let nap = info.part == Part::Nap;
        // During the day only presence counts (moving curtains or light are not a nap);
        // without the scene model fall back to the motion-based idle detection.
        let occupied = match info.presence {
            Some(p) => p >= c.presence_threshold,
            None => !idle,
        };
        if nap {
            self.track_nap(t, occupied);
        } else if occupied {
            self.last_night_seen = t;
        }
        let c = &self.cfg;
        let interval = if nap && !occupied {
            // an empty room during the day is never part of the timelapse
            self.stats.idle += 1;
            0.0
        } else if idle {
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
                self.queue(b);
            }
        }
        self.release(t)
    }

    fn queue(&mut self, mut b: (FrameInfo, P)) {
        if b.0.part == Part::Nap && self.nap.as_ref().map(|n| n.night_like).unwrap_or(false) {
            b.0.part = Part::Night;
        }
        // Only a running, not yet long enough nap holds frames back. (A nap frame
        // queued after its nap ended belongs to a qualified one: the best
        // candidate of an unqualified nap is dropped when it ends.)
        let gated = b.0.part == Part::Nap && self.nap.as_ref().map(|n| !n.qualified).unwrap_or(false);
        if gated {
            self.gate.push(b);
        } else {
            self.pending.push_back(b);
        }
    }

    fn block(&mut self, t: f64) {
        let (before, after) = (self.cfg.before, self.cfg.after);
        self.blocked.push((t - before, t + after));
        if self.best.as_ref().map(|(b, _)| b.t >= t - before).unwrap_or(false) {
            self.best = None;
        }
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
                if f.part == Part::Nap {
                    self.stats.emitted_nap += 1;
                }
                out.push((f, p));
            }
        }
        out
    }

    /// End of the night: flush everything that is not blocked.
    pub fn finish(&mut self) -> Vec<(FrameInfo, P)> {
        if let Some(b) = self.best.take() {
            if self.budget >= 0.5 {
                self.queue(b);
            }
        }
        self.end_nap();
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
            nap_min: 300.0,
            nap_gap: 60.0,
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

    fn nap(t: f64, presence: f32) -> FrameInfo {
        FrameInfo { part: Part::Nap, presence: Some(presence), ..frame(t, 0.05) }
    }

    #[test]
    fn short_daytime_visit_is_not_a_nap() {
        let mut s = Selector::new(cfg(), vec![]);
        // 4 minutes somebody there, then an empty room for an hour
        let frames: Vec<_> = (0..3840).map(|i| nap(i as f64, if i < 240 { 0.9 } else { 0.1 })).collect();
        assert!(run(&mut s, frames).is_empty());
        assert!(s.stats.dropped_short > 0);
    }

    #[test]
    fn nap_is_kept_including_its_start() {
        let mut s = Selector::new(cfg(), vec![]);
        // empty room, a 30 minute nap (with a 30 s flicker), empty room again
        let frames: Vec<_> = (0..4200)
            .map(|i| {
                let present = (600..2400).contains(&i) && !(1000..1030).contains(&i);
                nap(i as f64, if present { 0.9 } else { 0.1 })
            })
            .collect();
        let out = run(&mut s, frames);
        assert!(out.iter().all(|&t| (600.0..2400.0).contains(&t)), "{out:?}");
        assert!(out.iter().any(|&t| t < 700.0), "the beginning of the nap is kept");
        assert!(out.len() >= 150, "{}", out.len());
        assert_eq!(s.stats.emitted_nap as usize, out.len());
    }

    #[test]
    fn sleep_running_over_the_night_window_is_kept() {
        let mut s = Selector::new(cfg(), vec![]);
        // 3 min of daytime presence right before the night, night, 3 min after it
        let mut frames: Vec<_> = (0..180).map(|i| nap(i as f64, 0.9)).collect();
        frames.extend((180..1800).map(|i| frame(i as f64, 0.05)));
        frames.extend((1800..1980).map(|i| nap(i as f64, 0.9)));
        frames.extend((1980..3000).map(|i| nap(i as f64, 0.1)));
        let out = run(&mut s, frames);
        assert!(out.iter().any(|&t| t < 180.0), "falling asleep before the window: {out:?}");
        assert!(out.iter().any(|&t| (1800.0..1980.0).contains(&t)), "waking up after it");
        assert_eq!(s.stats.dropped_short, 0);
        assert_eq!(s.stats.emitted_nap, 0, "both ends count as night");
    }

    #[test]
    fn last_candidate_of_a_nap_is_not_lost() {
        let mut s = Selector::new(cfg(), vec![]);
        // a qualified nap that ends in the middle of an output slot, then night
        let mut frames: Vec<_> = (0..605).map(|i| nap(i as f64, 0.9)).collect();
        frames.extend((605..700).map(|i| FrameInfo { presence: Some(0.1), ..frame(i as f64, 0.0) }));
        frames.extend((700..2000).map(|i| frame(i as f64, 0.05)));
        let out = run(&mut s, frames);
        let mut sorted = out.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(out, sorted, "frames stay in time order");
        assert!(s.gate.is_empty());
    }

    #[test]
    fn privacy_context_reaches_into_used_footage() {
        let mut s = Selector::new(cfg(), vec![]);
        // used footage until t=600, a detection in the context right after it
        let mut frames: Vec<_> = (0..600).map(|i| frame(i as f64, 0.05)).collect();
        frames.extend((600..640).map(|i| FrameInfo {
            part: Part::Context,
            private: if i == 610 { 2.0 } else { 0.0 },
            ..frame(i as f64, 0.05)
        }));
        let out = run(&mut s, frames);
        assert!(out.iter().all(|&t| t < 580.0), "{out:?}");
        assert!(!out.is_empty());
        // and before: a detection in leading context blocks the start
        let mut s = Selector::new(cfg(), vec![]);
        let mut frames = vec![FrameInfo { part: Part::Context, private: 2.0, ..frame(0.0, 0.0) }];
        frames.extend((1..600).map(|i| frame(i as f64, 0.05)));
        let out = run(&mut s, frames);
        assert!(out.iter().all(|&t| t > 60.0), "{out:?}");
    }

    #[test]
    fn best_candidate_wins() {
        let mut s = Selector::new(cfg(), vec![]);
        let frames: Vec<_> = (0..10).map(|i| FrameInfo { sharpness: if i == 7 { 900.0 } else { 10.0 }, ..frame(i as f64, 0.05) }).collect();
        assert_eq!(run(&mut s, frames), vec![7.0]);
    }
}
