//! Cheap image measurements and model input preparation straight from
//! yuv420p frames (BT.709, limited range – what the decoder produces).

/// A borrowed yuv420p frame.
#[derive(Clone, Copy)]
pub struct Yuv<'a> {
    pub data: &'a [u8],
    pub w: usize,
    pub h: usize,
}

impl<'a> Yuv<'a> {
    pub fn new(data: &'a [u8], w: usize, h: usize) -> Self {
        debug_assert!(data.len() >= w * h * 3 / 2);
        Self { data, w, h }
    }

    /// RGB (0..1) of a pixel.
    #[inline]
    pub fn rgb(&self, x: usize, y: usize) -> [f32; 3] {
        let cw = self.w / 2;
        let ys = self.w * self.h;
        let ci = (y / 2) * cw + x / 2;
        let yy = (self.data[y * self.w + x] as f32 - 16.0) * (1.0 / 219.0);
        let u = (self.data[ys + ci] as f32 - 128.0) * (1.0 / 224.0);
        let v = (self.data[ys + ys / 4 + ci] as f32 - 128.0) * (1.0 / 224.0);
        // BT.709
        let r = yy + 1.5748 * v;
        let g = yy - 0.1873 * u - 0.4681 * v;
        let b = yy + 1.8556 * u;
        [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
    }

    /// Bilinear RGB sample at continuous source coordinates (pixel centers at .5).
    #[inline]
    fn rgb_bilinear(&self, sx: f32, sy: f32) -> [f32; 3] {
        let x = (sx - 0.5).max(0.0);
        let y = (sy - 0.5).max(0.0);
        let x0 = (x as usize).min(self.w - 1);
        let y0 = (y as usize).min(self.h - 1);
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let p00 = self.rgb(x0, y0);
        let p10 = self.rgb(x1, y0);
        let p01 = self.rgb(x0, y1);
        let p11 = self.rgb(x1, y1);
        let mut out = [0f32; 3];
        for c in 0..3 {
            let top = p00[c] + (p10[c] - p00[c]) * fx;
            let bot = p01[c] + (p11[c] - p01[c]) * fx;
            out[c] = top + (bot - top) * fy;
        }
        out
    }
}

/// Area-averaged luma thumbnail.
pub fn luma_thumb(f: &Yuv, tw: usize, th: usize) -> Vec<u8> {
    let mut out = vec![0u8; tw * th];
    for ty in 0..th {
        let y0 = ty * f.h / th;
        let y1 = ((ty + 1) * f.h / th).max(y0 + 1);
        for tx in 0..tw {
            let x0 = tx * f.w / tw;
            let x1 = ((tx + 1) * f.w / tw).max(x0 + 1);
            let mut sum = 0u32;
            for y in y0..y1 {
                let row = &f.data[y * f.w + x0..y * f.w + x1];
                sum += row.iter().map(|&p| p as u32).sum::<u32>();
            }
            out[ty * tw + tx] = (sum / ((y1 - y0) * (x1 - x0)) as u32) as u8;
        }
    }
    out
}

pub fn mean(v: &[u8]) -> f64 {
    v.iter().map(|&p| p as f64).sum::<f64>() / v.len().max(1) as f64
}

/// Fraction of thumbnail pixels that changed by more than `thr`, after
/// compensating a global brightness change (auto exposure, IR switching).
pub fn change_fraction(a: &[u8], b: &[u8], thr: u8) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }
    let shift = (mean(b) - mean(a)).round() as i32;
    let changed = a.iter().zip(b).filter(|(&x, &y)| ((y as i32 - shift) - x as i32).unsigned_abs() > thr as u32).count();
    changed as f64 / a.len() as f64
}

/// Variance of the Laplacian – higher = sharper (less motion blur).
pub fn sharpness(thumb: &[u8], w: usize, h: usize) -> f64 {
    if w < 3 || h < 3 {
        return 0.0;
    }
    let mut sum = 0f64;
    let mut sum2 = 0f64;
    let mut n = 0f64;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = thumb[y * w + x] as f64;
            let l =
                thumb[y * w + x - 1] as f64 + thumb[y * w + x + 1] as f64 + thumb[(y - 1) * w + x] as f64 + thumb[(y + 1) * w + x] as f64
                    - 4.0 * c;
            sum += l;
            sum2 += l * l;
            n += 1.0;
        }
    }
    let m = sum / n;
    sum2 / n - m * m
}

/// NudeNet input: the frame padded to a square at the bottom/right with black,
/// resized bilinearly to `size`x`size`, RGB, NCHW, 0..1 (cv2.dnn.blobFromImage).
pub fn nudenet_input(f: &Yuv, size: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(3 * size * size, 0.0);
    let side = f.w.max(f.h) as f32;
    let scale = side / size as f32;
    let plane = size * size;
    for oy in 0..size {
        let sy = (oy as f32 + 0.5) * scale;
        for ox in 0..size {
            let sx = (ox as f32 + 0.5) * scale;
            let px = if sx - 0.5 > f.w as f32 - 1.0 || sy - 0.5 > f.h as f32 - 1.0 {
                // inside the black padding (bilinear blending at the very edge is negligible)
                [0.0; 3]
            } else {
                f.rgb_bilinear(sx, sy)
            };
            let i = oy * size + ox;
            out[i] = px[0];
            out[plane + i] = px[1];
            out[2 * plane + i] = px[2];
        }
    }
}

pub const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Antialiased bicubic weights (PIL's `Image.resize(..., BICUBIC)`): for each
/// output pixel the first source index and the normalized tap weights.
fn bicubic_weights(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    fn cubic(x: f32) -> f32 {
        let a = -0.5f32;
        let x = x.abs();
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    }
    let scale = src as f32 / dst as f32;
    let fscale = scale.max(1.0);
    let support = 2.0 * fscale;
    (0..dst)
        .map(|i| {
            let center = (i as f32 + 0.5) * scale;
            let lo = ((center - support + 0.5).floor().max(0.0)) as usize;
            let hi = ((center + support + 0.5).floor() as usize).min(src);
            let mut w: Vec<f32> = (lo..hi).map(|j| cubic((j as f32 - center + 0.5) / fscale)).collect();
            let sum: f32 = w.iter().sum();
            if sum != 0.0 {
                w.iter_mut().for_each(|x| *x /= sum);
            }
            (lo, w)
        })
        .collect()
}

/// CLIP input like OpenAI's preprocessing: bicubic (antialiased) resize of the
/// short side to `size`, center crop, normalize, NCHW.
pub fn clip_input(f: &Yuv, size: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(3 * size * size, 0.0);
    let side = f.w.min(f.h);
    let x_off = (f.w - side) / 2;
    let y_off = (f.h - side) / 2;
    let wx = bicubic_weights(side, size);
    let wy = bicubic_weights(side, size);
    // horizontal pass over the rows of the crop: side x size x 3
    let mut tmp = vec![0f32; side * size * 3];
    let mut row = vec![[0f32; 3]; side];
    for y in 0..side {
        for (x, px) in row.iter_mut().enumerate() {
            *px = f.rgb(x_off + x, y_off + y);
        }
        for (ox, (lo, w)) in wx.iter().enumerate() {
            let mut acc = [0f32; 3];
            for (k, &wk) in w.iter().enumerate() {
                let p = row[lo + k];
                acc[0] += p[0] * wk;
                acc[1] += p[1] * wk;
                acc[2] += p[2] * wk;
            }
            let o = (y * size + ox) * 3;
            tmp[o..o + 3].copy_from_slice(&acc);
        }
    }
    let plane = size * size;
    for (oy, (lo, w)) in wy.iter().enumerate() {
        for ox in 0..size {
            let mut acc = [0f32; 3];
            for (k, &wk) in w.iter().enumerate() {
                let o = ((lo + k) * size + ox) * 3;
                acc[0] += tmp[o] * wk;
                acc[1] += tmp[o + 1] * wk;
                acc[2] += tmp[o + 2] * wk;
            }
            let i = oy * size + ox;
            for c in 0..3 {
                // PIL rounds to 8-bit after resizing
                let v = (acc[c] * 255.0).round().clamp(0.0, 255.0) / 255.0;
                out[c * plane + i] = (v - CLIP_MEAN[c]) / CLIP_STD[c];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: usize, h: usize, v: u8) -> Vec<u8> {
        let mut d = vec![v; w * h];
        d.extend(vec![128u8; w * h / 2]);
        d
    }

    #[test]
    fn motion_ignores_global_brightness() {
        let a = gray(64, 36, 80);
        let b = gray(64, 36, 110);
        let ta = luma_thumb(&Yuv::new(&a, 64, 36), 16, 9);
        let tb = luma_thumb(&Yuv::new(&b, 64, 36), 16, 9);
        assert_eq!(change_fraction(&ta, &tb, 10), 0.0);
        let mut c = a.clone();
        for y in 0..18 {
            for x in 0..32 {
                c[y * 64 + x] = 200;
            }
        }
        let tc = luma_thumb(&Yuv::new(&c, 64, 36), 16, 9);
        assert!(change_fraction(&ta, &tc, 10) > 0.2);
    }

    #[test]
    fn color_conversion() {
        let d = gray(4, 4, 235);
        let p = Yuv::new(&d, 4, 4).rgb(1, 1);
        assert!(p.iter().all(|&c| (c - 1.0).abs() < 1e-3));
        let mut out = Vec::new();
        nudenet_input(&Yuv::new(&gray(16, 8, 16), 16, 8), 8, &mut out);
        assert_eq!(out.len(), 192);
        clip_input(&Yuv::new(&gray(16, 8, 126), 16, 8), 4, &mut out);
        assert_eq!(out.len(), 48);
    }
}
