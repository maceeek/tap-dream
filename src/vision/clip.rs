//! CLIP (ViT-B/32) zero-shot scene understanding: is anybody in the picture,
//! how "interesting" is the moment, and does it look like breastfeeding.
//! Works with the OpenAI CLIP ONNX exports (clip-as-service `visual.onnx` /
//! `textual.onnx`) and the Hugging Face ones (`vision_model.onnx` / `text_model.onnx`).

use super::image::{clip_input, Yuv};
use super::onnx::Runtime;
use super::tokenizer::{ClipTokenizer, CONTEXT_LENGTH};
use anyhow::{anyhow, bail, Result};
use ort::session::Session;
use ort::value::TensorElementType;
use ort::value::{DynValue, Tensor};
use std::path::Path;

#[derive(Debug, Clone, Copy, Default)]
pub struct SceneScores {
    /// Probability that somebody (baby, parents) is in the picture.
    pub presence: f32,
    /// Probability mass of the "interesting moment" prompts.
    pub interest: f32,
    /// Probability mass of the private prompts (e.g. breastfeeding).
    pub private: f32,
}

pub struct PromptGroups<'a> {
    pub presence: &'a [String],
    pub empty: &'a [String],
    pub interest: &'a [String],
    pub neutral: &'a [String],
    pub private: &'a [String],
    pub contrast: &'a [String],
}

pub struct Clip {
    visual: Session,
    visual_input: String,
    size: usize,
    buf: Vec<f32>,
    presence: Vec<Vec<f32>>,
    empty: Vec<Vec<f32>>,
    interest: Vec<Vec<f32>>,
    neutral: Vec<Vec<f32>>,
    private: Vec<Vec<f32>>,
    contrast: Vec<Vec<f32>>,
    pub device: String,
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}

fn pick_embedding_output(outputs: &ort::session::SessionOutputs, preferred: &str) -> Result<Vec<f32>> {
    let value = outputs.get(preferred).unwrap_or(&outputs[0]);
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
    if shape.len() != 2 {
        bail!("CLIP output {preferred} has shape {shape:?}, expected [batch, dim]");
    }
    let mut v = data[..shape[1] as usize].to_vec();
    normalize(&mut v);
    Ok(v)
}

fn int_tensor(ty: Option<TensorElementType>, data: Vec<i64>) -> Result<DynValue> {
    let shape = vec![1i64, data.len() as i64];
    Ok(match ty {
        Some(TensorElementType::Int32) => {
            Tensor::from_array((shape, data.into_iter().map(|x| x as i32).collect::<Vec<_>>())).map_err(|e| anyhow!("{e}"))?.into_dyn()
        }
        _ => Tensor::from_array((shape, data)).map_err(|e| anyhow!("{e}"))?.into_dyn(),
    })
}

fn embed_texts(textual: &mut Session, tok: &ClipTokenizer, prompts: &[String]) -> Result<Vec<Vec<f32>>> {
    let ids_input = textual
        .inputs()
        .iter()
        .find(|i| i.name().contains("input_ids"))
        .or_else(|| textual.inputs().first())
        .ok_or_else(|| anyhow!("text model has no inputs"))?;
    let ids_name = ids_input.name().to_string();
    let ids_ty = ids_input.dtype().tensor_type();
    let mask =
        textual.inputs().iter().find(|i| i.name().contains("attention_mask")).map(|i| (i.name().to_string(), i.dtype().tensor_type()));
    let mut out = Vec::new();
    for p in prompts {
        let (ids, m) = tok.tokenize(p)?;
        debug_assert_eq!(ids.len(), CONTEXT_LENGTH);
        let mut inputs: Vec<(String, DynValue)> = vec![(ids_name.clone(), int_tensor(ids_ty, ids)?)];
        if let Some((name, ty)) = &mask {
            inputs.push((name.clone(), int_tensor(*ty, m)?));
        }
        let outputs = textual.run(inputs).map_err(|e| anyhow!("CLIP text inference failed: {e}"))?;
        out.push(pick_embedding_output(&outputs, "text_embeds")?);
    }
    Ok(out)
}

impl Clip {
    pub fn load(rt: &Runtime, visual_path: &Path, textual_path: &Path, prompts: &PromptGroups) -> Result<Self> {
        let tok = ClipTokenizer::new()?;
        let (mut textual, _) = rt.session(textual_path)?;
        let presence = embed_texts(&mut textual, &tok, prompts.presence)?;
        let empty = embed_texts(&mut textual, &tok, prompts.empty)?;
        let interest = embed_texts(&mut textual, &tok, prompts.interest)?;
        let neutral = embed_texts(&mut textual, &tok, prompts.neutral)?;
        let private = embed_texts(&mut textual, &tok, prompts.private)?;
        let contrast = embed_texts(&mut textual, &tok, prompts.contrast)?;
        drop(textual); // only the image tower is needed from now on
        let (visual, device) = rt.session(visual_path)?;
        let input = visual.inputs().first().ok_or_else(|| anyhow!("CLIP image model has no input"))?;
        let visual_input = input.name().to_string();
        let size = input.dtype().tensor_shape().and_then(|s| s.get(2).copied()).filter(|&d| d > 0).unwrap_or(224) as usize;
        Ok(Self { visual, visual_input, size, buf: Vec::new(), presence, empty, interest, neutral, private, contrast, device })
    }

    pub fn embed(&mut self, frame: &Yuv) -> Result<Vec<f32>> {
        clip_input(frame, self.size, &mut self.buf);
        let s = self.size as i64;
        let tensor = Tensor::from_array((vec![1i64, 3, s, s], std::mem::take(&mut self.buf))).map_err(|e| anyhow!("{e}"))?;
        let outputs =
            self.visual.run(ort::inputs![self.visual_input.as_str() => tensor]).map_err(|e| anyhow!("CLIP image inference failed: {e}"))?;
        pick_embedding_output(&outputs, "image_embeds")
    }

    pub fn scores(&mut self, frame: &Yuv) -> Result<SceneScores> {
        let img = self.embed(frame)?;
        Ok(SceneScores {
            presence: group_probability(&img, &self.presence, &[&self.empty]),
            interest: group_probability(&img, &self.interest, &[&self.neutral]),
            private: if self.private.is_empty() {
                0.0
            } else {
                group_probability(&img, &self.private, &[&self.contrast, &self.neutral, &self.presence, &self.empty])
            },
        })
    }
}

/// Softmax (CLIP logit scale 100) over `target` ∪ `others`; returns the mass of `target`.
fn group_probability(img: &[f32], target: &[Vec<f32>], others: &[&Vec<Vec<f32>>]) -> f32 {
    if target.is_empty() {
        return 0.0;
    }
    let logit = |t: &Vec<f32>| 100.0 * t.iter().zip(img).map(|(a, b)| a * b).sum::<f32>();
    let t: Vec<f32> = target.iter().map(logit).collect();
    let o: Vec<f32> = others.iter().flat_map(|g| g.iter().map(logit)).collect();
    let max = t.iter().chain(o.iter()).copied().fold(f32::MIN, f32::max);
    let et: f32 = t.iter().map(|x| (x - max).exp()).sum();
    let eo: f32 = o.iter().map(|x| (x - max).exp()).sum();
    et / (et + eo)
}
