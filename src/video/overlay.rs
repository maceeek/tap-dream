//! Burns a small caption ("Noc 123 · 13. 10. 2026") into yuv420p frames.

use ab_glyph::{Font, FontVec, PxScale, ScaleFont};
use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct Overlay {
    font: FontVec,
    px: f32,
    position: String,
    cache: HashMap<String, Mask>,
}

struct Mask {
    w: usize,
    h: usize,
    alpha: Vec<f32>,
}

const FONT_CANDIDATES: &[&str] = &[
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
];

pub fn find_font(configured: &str) -> Option<PathBuf> {
    if !configured.is_empty() {
        return Some(PathBuf::from(configured));
    }
    for c in FONT_CANDIDATES {
        if Path::new(c).exists() {
            return Some(PathBuf::from(c));
        }
    }
    let o = std::process::Command::new("fc-match").args(["-f", "%{file}", "sans:lang=cs"]).output().ok()?;
    let p = PathBuf::from(String::from_utf8_lossy(&o.stdout).trim());
    p.exists().then_some(p)
}

impl Overlay {
    pub fn new(font_path: &Path, frame_height: u32, size: f32, position: &str) -> Result<Self> {
        let data = std::fs::read(font_path).with_context(|| format!("cannot read font {}", font_path.display()))?;
        let font = FontVec::try_from_vec(data).map_err(|_| anyhow!("{} is not a usable font", font_path.display()))?;
        let px = (frame_height as f32 * size).max(10.0);
        Ok(Self { font, px, position: position.to_string(), cache: HashMap::new() })
    }

    fn render(&self, text: &str) -> Mask {
        let scale = PxScale::from(self.px);
        let sf = self.font.as_scaled(scale);
        let mut x = 0.0f32;
        let mut glyphs = Vec::new();
        let mut prev = None;
        for ch in text.chars() {
            let id = sf.glyph_id(ch);
            if let Some(p) = prev {
                x += sf.kern(p, id);
            }
            glyphs.push(id.with_scale_and_position(scale, ab_glyph::point(x, sf.ascent())));
            x += sf.h_advance(id);
            prev = Some(id);
        }
        let pad = (self.px * 0.12).ceil() as usize + 2;
        let w = x.ceil() as usize + pad * 2;
        let h = (sf.ascent() - sf.descent()).ceil() as usize + pad * 2;
        let mut glyph_alpha = vec![0f32; w * h];
        for g in glyphs {
            if let Some(o) = self.font.outline_glyph(g) {
                let b = o.px_bounds();
                o.draw(|gx, gy, c| {
                    let px = b.min.x as i64 + gx as i64 + pad as i64;
                    let py = b.min.y as i64 + gy as i64 + pad as i64;
                    if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                        let i = py as usize * w + px as usize;
                        glyph_alpha[i] = glyph_alpha[i].max(c);
                    }
                });
            }
        }
        // Encode text (positive) and a soft dark halo (negative) in one mask.
        let r = ((self.px * 0.08).round() as i64).max(1);
        let mut alpha = vec![0f32; w * h];
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let mut halo = 0f32;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (sx, sy) = (x + dx, y + dy);
                        if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
                            halo = halo.max(glyph_alpha[sy as usize * w + sx as usize]);
                        }
                    }
                }
                let a = glyph_alpha[y as usize * w + x as usize];
                alpha[y as usize * w + x as usize] = if a > 0.0 { a } else { -halo * 0.65 };
            }
        }
        Mask { w, h, alpha }
    }

    /// Draws `text` onto a yuv420p frame of the given size.
    pub fn draw(&mut self, frame: &mut [u8], width: usize, height: usize, text: &str) {
        if text.is_empty() {
            return;
        }
        if !self.cache.contains_key(text) {
            if self.cache.len() > 64 {
                self.cache.clear();
            }
            let m = self.render(text);
            self.cache.insert(text.to_string(), m);
        }
        let m = &self.cache[text];
        let margin = (height as f32 * 0.03) as usize;
        let (ox, oy) = match self.position.as_str() {
            "top-left" => (margin, margin),
            "top-right" => (width.saturating_sub(m.w + margin), margin),
            "bottom-right" => (width.saturating_sub(m.w + margin), height.saturating_sub(m.h + margin)),
            _ => (margin, height.saturating_sub(m.h + margin)),
        };
        let (y_plane, chroma) = frame.split_at_mut(width * height);
        let (u_plane, v_plane) = chroma.split_at_mut(width * height / 4);
        let cw = width / 2;
        for my in 0..m.h {
            let fy = oy + my;
            if fy >= height {
                break;
            }
            for mx in 0..m.w {
                let fx = ox + mx;
                if fx >= width {
                    break;
                }
                let a = m.alpha[my * m.w + mx];
                if a == 0.0 {
                    continue;
                }
                let yi = fy * width + fx;
                let (target, k) = if a > 0.0 { (235.0, a) } else { (16.0, -a) };
                y_plane[yi] = (y_plane[yi] as f32 * (1.0 - k) + target * k).round() as u8;
                if fx % 2 == 0 && fy % 2 == 0 {
                    let ci = (fy / 2) * cw + fx / 2;
                    u_plane[ci] = (u_plane[ci] as f32 * (1.0 - k) + 128.0 * k).round() as u8;
                    v_plane[ci] = (v_plane[ci] as f32 * (1.0 - k) + 128.0 * k).round() as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_text() {
        let Some(font) = find_font("") else { return };
        let (w, h) = (320usize, 180usize);
        let mut frame = vec![128u8; w * h * 3 / 2];
        let mut o = Overlay::new(&font, h as u32, 0.1, "bottom-left").unwrap();
        o.draw(&mut frame, w, h, "Noc 12 · 1. 2. 2026");
        let bright = frame[..w * h].iter().filter(|&&p| p > 200).count();
        let dark = frame[..w * h].iter().filter(|&&p| p < 100).count();
        assert!(bright > 100 && dark > 100, "{bright} {dark}");
        // nothing drawn in the top half
        assert!(frame[..w * h / 2].iter().all(|&p| p == 128));
    }
}
