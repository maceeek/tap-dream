//! OpenAI CLIP byte-level BPE tokenizer (port of `simple_tokenizer.py`,
//! MIT, OpenAI). The 1.3 MB merge list is embedded in the binary.

use anyhow::{anyhow, Result};
use regex::Regex;
use std::collections::HashMap;
use std::io::Read;

static BPE_GZ: &[u8] = include_bytes!("../../assets/bpe_simple_vocab_16e6.txt.gz");
pub const CONTEXT_LENGTH: usize = 77;

pub struct ClipTokenizer {
    encoder: HashMap<String, i64>,
    ranks: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    pat: Regex,
    pub sot: i64,
    pub eot: i64,
}

/// (byte -> char table, chars in vocabulary order)
fn bytes_to_unicode() -> ([char; 256], Vec<char>) {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    let mut order = Vec::with_capacity(256);
    for (b, c) in bs.into_iter().zip(cs) {
        let ch = char::from_u32(c).unwrap();
        out[b as usize] = ch;
        order.push(ch);
    }
    (out, order)
}

impl ClipTokenizer {
    pub fn new() -> Result<Self> {
        let mut text = String::new();
        flate2::read::GzDecoder::new(BPE_GZ).read_to_string(&mut text)?;
        let merges: Vec<(String, String)> = text
            .split('\n')
            .skip(1)
            .take(49152 - 256 - 2)
            .map(|l| {
                let mut it = l.split(' ');
                (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string())
            })
            .collect();
        let (byte_encoder, order) = bytes_to_unicode();
        let mut vocab: Vec<String> = order.iter().map(|c| c.to_string()).collect();
        let with_w: Vec<String> = vocab.iter().map(|v| format!("{v}</w>")).collect();
        vocab.extend(with_w);
        for (a, b) in &merges {
            vocab.push(format!("{a}{b}"));
        }
        vocab.push("<|startoftext|>".into());
        vocab.push("<|endoftext|>".into());
        let encoder: HashMap<String, i64> = vocab.into_iter().enumerate().map(|(i, v)| (v, i as i64)).collect();
        let ranks = merges.into_iter().enumerate().map(|(i, m)| (m, i)).collect();
        let pat = Regex::new(r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+")?;
        let sot = encoder["<|startoftext|>"];
        let eot = encoder["<|endoftext|>"];
        Ok(Self { encoder, ranks, byte_encoder, pat, sot, eot })
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let chars: Vec<char> = token.chars().collect();
        let mut word: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        if word.len() < 2 {
            return word;
        }
        loop {
            let best = word
                .windows(2)
                .filter_map(|p| self.ranks.get(&(p[0].clone(), p[1].clone())).map(|r| (*r, p[0].clone(), p[1].clone())))
                .min_by_key(|(r, _, _)| *r);
            let Some((_, first, second)) = best else { break };
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    merged.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
            if word.len() == 1 {
                break;
            }
        }
        word
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let clean = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let mut ids = Vec::new();
        for m in self.pat.find_iter(&clean) {
            let t = m.as_str();
            if t == "<|startoftext|>" || t == "<|endoftext|>" {
                ids.push(self.encoder[t]);
                continue;
            }
            let mapped: String = t.bytes().map(|b| self.byte_encoder[b as usize]).collect();
            for piece in self.bpe(&mapped) {
                ids.push(*self.encoder.get(&piece).ok_or_else(|| anyhow!("token {piece:?} not in vocabulary"))?);
            }
        }
        Ok(ids)
    }

    /// `[sot] + tokens + [eot]`, truncated and zero-padded to 77; plus the attention mask.
    pub fn tokenize(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let mut ids = vec![self.sot];
        ids.extend(self.encode(text)?);
        ids.truncate(CONTEXT_LENGTH - 1);
        ids.push(self.eot);
        let mut mask = vec![1i64; ids.len()];
        ids.resize(CONTEXT_LENGTH, 0);
        mask.resize(CONTEXT_LENGTH, 0);
        Ok((ids, mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference ids from open_clip's SimpleTokenizer (see tools/gen_test_vectors.py).
    #[test]
    fn matches_open_clip() {
        let t = ClipTokenizer::new().unwrap();
        let v: Vec<(String, Vec<i64>)> = serde_json::from_str(include_str!("testdata/clip_tokens.json")).unwrap();
        for (text, ids) in v {
            let (got, _) = t.tokenize(&text).unwrap();
            assert_eq!(got, ids, "{text}");
        }
    }
}
