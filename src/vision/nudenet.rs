//! NudeNet v3 detector (YOLOv8, ONNX) – flags frames with exposed body parts.
//! Preprocessing mirrors nudenet.py: pad to square (bottom/right), resize,
//! RGB, 0..1. The highest class score over all anchors is what we use.

use super::image::{nudenet_input, Yuv};
use super::onnx::Runtime;
use anyhow::{anyhow, bail, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

pub const LABELS: [&str; 18] = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
];

pub struct NudeNet {
    session: Session,
    input_name: String,
    size: usize,
    watched: Vec<usize>,
    buf: Vec<f32>,
    pub device: String,
}

impl NudeNet {
    pub fn load(rt: &Runtime, path: &Path, input_size: u32, classes: &[String]) -> Result<Self> {
        let (session, device) = rt.session(path)?;
        let input = session.inputs().first().ok_or_else(|| anyhow!("NudeNet model has no input"))?;
        let input_name = input.name().to_string();
        let fixed = input.dtype().tensor_shape().and_then(|s| s.get(2).copied()).filter(|&d| d > 0).map(|d| d as usize);
        let size = if input_size > 0 {
            input_size as usize
        } else if let Some(f) = fixed {
            f
        } else if path.to_string_lossy().contains("640") {
            640
        } else {
            320
        };
        let mut watched = Vec::new();
        for c in classes {
            match LABELS.iter().position(|l| l.eq_ignore_ascii_case(c)) {
                Some(i) => watched.push(i),
                None => bail!("unknown NudeNet class {c:?}; known: {}", LABELS.join(", ")),
            }
        }
        if watched.is_empty() {
            bail!("privacy.classes is empty");
        }
        Ok(Self { session, input_name, size, watched, buf: Vec::new(), device })
    }

    /// Highest score of every class anywhere in the frame.
    pub fn class_scores(&mut self, frame: &Yuv) -> Result<[f32; 18]> {
        nudenet_input(frame, self.size, &mut self.buf);
        let s = self.size as i64;
        let tensor = Tensor::from_array((vec![1i64, 3, s, s], std::mem::take(&mut self.buf))).map_err(|e| anyhow!("{e}"))?;
        let outputs =
            self.session.run(ort::inputs![self.input_name.as_str() => tensor]).map_err(|e| anyhow!("NudeNet inference failed: {e}"))?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        if shape.len() != 3 || shape[1] < 22 {
            bail!("unexpected NudeNet output shape {shape:?}");
        }
        let anchors = shape[2] as usize;
        let mut best = [0f32; 18];
        for (c, b) in best.iter_mut().enumerate() {
            let row = &data[(4 + c) * anchors..(5 + c) * anchors];
            *b = row.iter().copied().fold(0.0, f32::max);
        }
        Ok(best)
    }

    /// The most confident of the watched (private) classes.
    pub fn private_score(&mut self, frame: &Yuv) -> Result<(f32, &'static str)> {
        let s = self.class_scores(frame)?;
        let (i, v) = self.watched.iter().map(|&i| (i, s[i])).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        Ok((v, LABELS[i]))
    }
}
