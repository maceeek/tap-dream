//! ffmpeg/ffprobe subprocesses: hardware-accelerated decoding of recordings
//! into raw frames, and encoding of selected frames into fragmented MP4
//! segments that can be appended to the timelapse.

use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

/// Track timescale of all segments; divisible by every common frame rate.
pub const TIMESCALE: u32 = 90_000;

#[derive(Debug, Clone, Default)]
pub struct Caps {
    pub hwaccels: HashSet<String>,
    pub encoders: HashSet<String>,
    pub nvidia: bool,
    pub version: String,
}

pub fn caps() -> Result<Caps> {
    let run = |args: &[&str]| -> Result<String> {
        let o = Command::new("ffmpeg").args(args).stdin(Stdio::null()).output().context("ffmpeg not found on PATH (install ffmpeg)")?;
        Ok(String::from_utf8_lossy(&o.stdout).to_string())
    };
    let version = run(&["-hide_banner", "-version"])?.lines().next().unwrap_or("").to_string();
    let hwaccels = run(&["-hide_banner", "-hwaccels"])?.lines().skip(1).map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
    let encoders = run(&["-hide_banner", "-encoders"])?
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let flags = it.next()?;
            let name = it.next()?;
            (flags.len() == 6 && flags.starts_with('V')).then(|| name.to_string())
        })
        .collect();
    let nvidia = Path::new("/dev/nvidia0").exists() || Path::new("/proc/driver/nvidia/version").exists();
    Ok(Caps { hwaccels, encoders, nvidia, version })
}

/// Median keyframe interval of the first minute of a recording (packet flags, no decoding).
pub fn keyframe_interval(path: &Path) -> Option<f64> {
    let o = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            "%+60",
            "-show_entries",
            "packet=pts_time,flags",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .ok()?;
    let mut keys: Vec<f64> = String::from_utf8_lossy(&o.stdout)
        .lines()
        .filter_map(|l| {
            let (t, f) = l.split_once(',')?;
            f.contains('K').then(|| t.parse::<f64>().ok()).flatten()
        })
        .collect();
    keys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if keys.len() < 3 {
        return None;
    }
    let mut d: Vec<f64> = keys.windows(2).map(|w| w[1] - w[0]).filter(|x| *x > 0.0).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d.get(d.len() / 2).copied()
}

fn spawn_stderr_collector(child: &mut Child) -> Arc<Mutex<String>> {
    let buf = Arc::new(Mutex::new(String::new()));
    if let Some(mut err) = child.stderr.take() {
        let b = buf.clone();
        std::thread::spawn(move || {
            let mut s = String::new();
            let _ = err.read_to_string(&mut s);
            let mut g = b.lock().unwrap();
            g.push_str(&s);
        });
    }
    buf
}

fn tail(s: &str) -> String {
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(6)..].join("\n")
}

#[derive(Clone, Debug)]
pub struct DecodeOptions {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub crop: String,
    pub cuda: bool,
    pub keyframes_only: bool,
    /// Seek offset into the file and length to decode, seconds.
    pub start: f64,
    pub length: Option<f64>,
    /// Single images: emit the frame even though it is shorter than 1/fps.
    pub still: bool,
}

/// Raw yuv420p frames at a fixed rate from a recording.
pub struct Decoder {
    child: Child,
    out: Option<ChildStdout>,
    stderr: Arc<Mutex<String>>,
    pub frame_size: usize,
    pub frames_read: u64,
}

impl Decoder {
    pub fn spawn(input: &Path, o: &DecodeOptions) -> Result<Self> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-nostdin", "-hide_banner", "-loglevel", "error"]);
        if o.cuda {
            cmd.args(["-hwaccel", "cuda"]);
        }
        if o.keyframes_only {
            cmd.args(["-skip_frame", "nokey"]);
        }
        if o.start > 0.0 {
            cmd.args(["-ss", &format!("{:.3}", o.start)]);
        }
        cmd.arg("-i").arg(input);
        if let Some(l) = o.length {
            cmd.args(["-t", &format!("{:.3}", l)]);
        }
        let mut vf = format!("fps={}{}", o.fps, if o.still { ":eof_action=pass" } else { "" });
        if !o.crop.trim().is_empty() {
            vf.push_str(&format!(",crop={}", o.crop.trim()));
        }
        vf.push_str(&format!(
            ",scale={w}:{h}:force_original_aspect_ratio=decrease:flags=area:out_color_matrix=bt709:out_range=tv,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            w = o.width,
            h = o.height
        ));
        cmd.args(["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", &vf, "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]);
        cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = cmd.spawn().context("cannot start ffmpeg")?;
        let stderr = spawn_stderr_collector(&mut child);
        let out = child.stdout.take();
        Ok(Self { child, out, stderr, frame_size: (o.width * o.height * 3 / 2) as usize, frames_read: 0 })
    }

    /// Fills `buf` with the next frame. Returns false at the end of the stream.
    pub fn read_frame(&mut self, buf: &mut Vec<u8>) -> Result<bool> {
        buf.resize(self.frame_size, 0);
        let mut filled = 0;
        let out = self.out.as_mut().ok_or_else(|| anyhow!("decoder closed"))?;
        while filled < self.frame_size {
            let n = out.read(&mut buf[filled..])?;
            if n == 0 {
                if filled > 0 {
                    log::debug!("decoder ended with a partial frame");
                }
                return Ok(false);
            }
            filled += n;
        }
        self.frames_read += 1;
        Ok(true)
    }

    /// Waits for ffmpeg; errors only if it failed without producing anything.
    pub fn finish(mut self) -> Result<()> {
        drop(self.out.take());
        let status = self.child.wait()?;
        std::thread::sleep(std::time::Duration::from_millis(20));
        let err = self.stderr.lock().unwrap().clone();
        if !status.success() && self.frames_read == 0 {
            bail!("ffmpeg could not decode the recording: {}", tail(&err));
        }
        if !err.trim().is_empty() {
            log::debug!("ffmpeg decoder: {}", tail(&err));
        }
        Ok(())
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderSpec {
    /// ffmpeg encoder name, e.g. hevc_nvenc, libx265, h264_nvenc, libx264
    pub name: String,
    pub quality: u32,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl EncoderSpec {
    /// Picks an encoder for the configured codec ("hevc"/"h264") and preference.
    pub fn choose(caps: &Caps, codec: &str, preference: &str, quality: u32, width: u32, height: u32, fps: u32) -> Result<Self> {
        let (nv, sw) = match codec {
            "h264" => ("h264_nvenc", "libx264"),
            _ => ("hevc_nvenc", "libx265"),
        };
        let name = match preference {
            "nvenc" => {
                if !caps.encoders.contains(nv) {
                    bail!("{nv} is not available in this ffmpeg build");
                }
                nv
            }
            "software" => sw,
            _ => {
                if caps.nvidia && caps.encoders.contains(nv) && nvenc_works(nv) {
                    nv
                } else {
                    sw
                }
            }
        };
        if !caps.encoders.contains(name) {
            bail!("encoder {name} is not available in this ffmpeg build");
        }
        Ok(Self { name: name.to_string(), quality, width, height, fps })
    }

    fn codec_args(&self) -> Vec<String> {
        let q = self.quality.to_string();
        let gop = (self.fps * 2).to_string();
        let mut a: Vec<String> = match self.name.as_str() {
            "hevc_nvenc" | "h264_nvenc" => vec!["-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", &q, "-b:v", "0", "-spatial-aq", "1"]
                .into_iter()
                .map(String::from)
                .collect(),
            "libx265" => vec!["-preset", "medium", "-crf", &q, "-x265-params", "log-level=error"].into_iter().map(String::from).collect(),
            _ => vec!["-preset", "slow", "-crf", &q].into_iter().map(String::from).collect(),
        };
        if self.name.starts_with("hevc") || self.name == "libx265" {
            a.extend(["-tag:v", "hvc1", "-profile:v", "main"].map(String::from));
        } else {
            a.extend(["-tag:v", "avc1", "-profile:v", "high"].map(String::from));
        }
        a.extend(["-g", &gop, "-bf", "0", "-pix_fmt", "yuv420p"].map(String::from));
        a
    }
}

fn nvenc_works(name: &str) -> bool {
    // NVENC may be compiled in but unusable (no driver/GPU, session limit).
    Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=size=256x256:rate=1:duration=1",
            "-frames:v",
            "1",
            "-c:v",
            name,
            "-f",
            "null",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Encodes raw yuv420p frames into an appendable fragmented MP4.
pub struct Encoder {
    child: Child,
    stdin: Option<ChildStdin>,
    stderr: Arc<Mutex<String>>,
    frame_size: usize,
    pub frames: u64,
}

impl Encoder {
    pub fn spawn(spec: &EncoderSpec, output: &Path) -> Result<Self> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y"]);
        cmd.args([
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            &format!("{}x{}", spec.width, spec.height),
            "-r",
            &spec.fps.to_string(),
            "-i",
            "pipe:0",
        ]);
        cmd.args(["-an", "-c:v", &spec.name]);
        cmd.args(spec.codec_args());
        cmd.args(["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]);
        // Parameter sets in every keyframe keep later segments decodable even
        // if an encoder update changes them; fragments are self-contained.
        cmd.args(["-bsf:v", "dump_extra=freq=keyframe"]);
        cmd.args(["-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-video_track_timescale", &TIMESCALE.to_string()]);
        cmd.args(["-metadata", "comment=tapdream timelapse", "-f", "mp4"]);
        cmd.arg(output);
        cmd.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());
        let mut child = cmd.spawn().context("cannot start ffmpeg encoder")?;
        let stderr = spawn_stderr_collector(&mut child);
        let stdin = child.stdin.take();
        Ok(Self { child, stdin, stderr, frame_size: (spec.width * spec.height * 3 / 2) as usize, frames: 0 })
    }

    pub fn write_frame(&mut self, yuv: &[u8]) -> Result<()> {
        if yuv.len() != self.frame_size {
            bail!("frame has {} bytes, expected {}", yuv.len(), self.frame_size);
        }
        let stdin = self.stdin.as_mut().ok_or_else(|| anyhow!("encoder closed"))?;
        if let Err(e) = stdin.write_all(yuv) {
            let err = self.stderr.lock().unwrap().clone();
            bail!("encoder failed: {e} {}", tail(&err));
        }
        self.frames += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        drop(self.stdin.take());
        let status = self.child.wait()?;
        std::thread::sleep(std::time::Duration::from_millis(20));
        if !status.success() {
            let err = self.stderr.lock().unwrap().clone();
            bail!("encoder failed: {}", tail(&err));
        }
        Ok(self.frames)
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        if self.stdin.is_some() {
            drop(self.stdin.take());
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// Writes one raw frame as a JPEG (highlights).
pub fn write_jpeg(yuv: &[u8], width: u32, height: u32, out: &Path) -> Result<()> {
    let mut child = Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s"])
        .arg(format!("{width}x{height}"))
        .args(["-i", "pipe:0", "-frames:v", "1", "-q:v", "2"])
        .arg(out)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    child.stdin.take().unwrap().write_all(yuv)?;
    if !child.wait()?.success() {
        bail!("cannot write {}", out.display());
    }
    Ok(())
}

/// Plain progressive MP4 copy of the timelapse (for sharing/archiving).
pub fn export(input: &Path, output: &Path) -> Result<()> {
    let st = Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(input)
        .args(["-c", "copy", "-movflags", "+faststart"])
        .arg(output)
        .status()?;
    if !st.success() {
        bail!("ffmpeg export failed");
    }
    Ok(())
}
