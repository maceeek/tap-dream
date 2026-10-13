//! Downloads the (public) AI models into data_dir/models. This is the only
//! moment the tool talks to the internet; all inference is local.

use anyhow::{bail, Context, Result};
use md5::{Digest, Md5};
use sha2::Sha256;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Model {
    file: &'static str,
    url: &'static str,
    /// md5 or sha256 hex of the file, when known
    md5: Option<&'static str>,
    sha256: Option<&'static str>,
    what: &'static str,
}

const MODELS: &[Model] = &[
    Model {
        file: "clip-vit-b32-visual.onnx",
        url: "https://clip-as-service.s3.us-east-2.amazonaws.com/models-436c69702d61732d53657276696365/onnx/ViT-B-32/visual.onnx",
        md5: Some("88c6f38e522269d6c04a85df18e6370c"),
        sha256: None,
        what: "CLIP ViT-B/32 image encoder (OpenAI, via clip-as-service)",
    },
    Model {
        file: "clip-vit-b32-textual.onnx",
        url: "https://clip-as-service.s3.us-east-2.amazonaws.com/models-436c69702d61732d53657276696365/onnx/ViT-B-32/textual.onnx",
        md5: Some("bd6d7871e8bb95f3cc83aff3398d7390"),
        sha256: None,
        what: "CLIP ViT-B/32 text encoder",
    },
    Model {
        file: "nudenet-640m.onnx",
        url: "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx",
        md5: None,
        sha256: None,
        what: "NudeNet 640m detector (more accurate, needs a GPU for speed)",
    },
];

const NUDENET_WHEEL: &str = "https://files.pythonhosted.org/packages/1c/ee/1aa02d44ba958cc77e16ff1e41a0aac5e721037db7bf62b9c9d124917f87/nudenet-3.4.2-py3-none-any.whl";
const NUDENET_WHEEL_SHA256: &str = "5937dbd84e5d8e5de038f08ffea5a1bb50a08475776bf2b4795914ce0eaf0331";
const NUDENET_320N_SHA256: &str = "c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f";

fn curl(url: &str, out: &Path) -> Result<()> {
    let st = Command::new("curl")
        .args(["-fL", "--retry", "3", "--progress-bar", "-o"])
        .arg(out)
        .arg(url)
        .status()
        .context("curl is needed to download models")?;
    if !st.success() {
        bail!("download of {url} failed");
    }
    Ok(())
}

fn hash_file<D: Digest>(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = D::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

fn verify(path: &Path, md5: Option<&str>, sha: Option<&str>) -> Result<()> {
    if let Some(m) = md5 {
        let got = hash_file::<Md5>(path)?;
        if got != m {
            bail!("{} has md5 {got}, expected {m}", path.display());
        }
    }
    if let Some(s) = sha {
        let got = hash_file::<Sha256>(path)?;
        if got != s {
            bail!("{} has sha256 {got}, expected {s}", path.display());
        }
    }
    Ok(())
}

/// NudeNet 320n ships inside the pip wheel – extract it from there.
fn fetch_320n(dir: &Path) -> Result<PathBuf> {
    let target = dir.join("nudenet-320n.onnx");
    if target.exists() && verify(&target, None, Some(NUDENET_320N_SHA256)).is_ok() {
        return Ok(target);
    }
    let wheel = dir.join("nudenet.whl.part");
    println!("  NudeNet 320n detector (from the nudenet 3.4.2 wheel)");
    curl(NUDENET_WHEEL, &wheel)?;
    verify(&wheel, None, Some(NUDENET_WHEEL_SHA256))?;
    let mut zip = zip::ZipArchive::new(std::fs::File::open(&wheel)?)?;
    let mut entry = zip.by_name("nudenet/320n.onnx")?;
    let tmp = dir.join("nudenet-320n.onnx.part");
    let mut out = std::fs::File::create(&tmp)?;
    std::io::copy(&mut entry, &mut out)?;
    drop(out);
    std::fs::remove_file(&wheel)?;
    verify(&tmp, None, Some(NUDENET_320N_SHA256))?;
    std::fs::rename(&tmp, &target)?;
    Ok(target)
}

pub fn fetch(models_dir: &Path, with_clip: bool) -> Result<()> {
    std::fs::create_dir_all(models_dir)?;
    fetch_320n(models_dir)?;
    for m in MODELS {
        if !with_clip && m.file.starts_with("clip") {
            continue;
        }
        let target = models_dir.join(m.file);
        if target.exists() && verify(&target, m.md5, m.sha256).is_ok() {
            println!("  {} – already present", m.file);
            continue;
        }
        println!("  {}", m.what);
        let tmp = models_dir.join(format!("{}.part", m.file));
        match curl(m.url, &tmp).and_then(|_| verify(&tmp, m.md5, m.sha256)) {
            Ok(()) => std::fs::rename(&tmp, &target)?,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                if m.file == "nudenet-640m.onnx" {
                    eprintln!("  could not get the 640m model ({e:#}); set privacy.model = \"models/nudenet-320n.onnx\"");
                } else {
                    return Err(e);
                }
            }
        }
    }
    Ok(())
}
