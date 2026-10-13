//! ONNX Runtime setup. The runtime library is loaded dynamically, so the same
//! binary runs with the GPU build (`onnxruntime-gpu`, CUDA/TensorRT) or the
//! CPU build. Everything runs locally; no model ever leaves the machine.

use crate::config::Gpu;
use anyhow::{anyhow, bail, Context, Result};
use ort::environment::Environment;
use ort::ep;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub struct Runtime {
    pub env: Environment,
    pub lib: PathBuf,
    pub provider: String,
    /// The provider was chosen automatically, so falling back to the CPU is fine.
    auto: bool,
    device_id: i32,
    cpu_threads: usize,
    cache_dir: PathBuf,
}

static RUNTIME: OnceLock<std::result::Result<Runtime, String>> = OnceLock::new();

fn glob_dir(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir).map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).collect()).unwrap_or_default();
    v.retain(|p| p.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with(prefix)).unwrap_or(false));
    v.sort();
    v
}

/// Python site-packages directories that may contain `onnxruntime`.
fn site_packages() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = vec![PathBuf::from("/usr/lib"), PathBuf::from("/usr/local/lib")];
    if let Ok(home) = std::env::var("HOME") {
        roots.push(PathBuf::from(&home).join(".local/lib"));
        for venv in [".tapdream-venv", ".venv", "venv"] {
            roots.push(PathBuf::from(&home).join(venv).join("lib"));
        }
    }
    if let Ok(v) = std::env::var("VIRTUAL_ENV") {
        roots.push(PathBuf::from(v).join("lib"));
    }
    let mut out = Vec::new();
    for r in roots {
        for py in glob_dir(&r, "python3") {
            for sp in ["site-packages", "dist-packages"] {
                let p = py.join(sp);
                if p.is_dir() {
                    out.push(p);
                }
            }
        }
    }
    out
}

pub fn find_library(cfg: &Gpu, data_dir: &Path) -> Option<PathBuf> {
    if !cfg.onnxruntime_lib.is_empty() {
        return Some(PathBuf::from(&cfg.onnxruntime_lib));
    }
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        return Some(PathBuf::from(p));
    }
    let mut candidates = vec![data_dir.join("onnxruntime")];
    candidates.extend(site_packages().into_iter().map(|p| p.join("onnxruntime").join("capi")));
    candidates.extend(
        ["/usr/local/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/opt/onnxruntime/lib"]
            .map(PathBuf::from),
    );
    // Prefer a GPU build (it ships the CUDA provider next to the runtime).
    let mut found: Vec<PathBuf> = Vec::new();
    for dir in candidates {
        for lib in glob_dir(&dir, "libonnxruntime.so") {
            if lib.file_name().and_then(|n| n.to_str()).map(|n| !n.contains("providers")).unwrap_or(false) {
                found.push(lib);
            }
        }
    }
    found.sort_by_key(|p| !p.parent().map(|d| d.join("libonnxruntime_providers_cuda.so").exists()).unwrap_or(false));
    found.into_iter().next()
}

/// The pip `nvidia-*-cu12` wheels put CUDA/cuDNN into site-packages/nvidia/*/lib.
/// Load them globally so the CUDA provider finds them without LD_LIBRARY_PATH.
fn preload_cuda_wheels(ort_lib: &Path) {
    let Some(sp) = ort_lib.ancestors().find(|p| p.ends_with("site-packages") || p.ends_with("dist-packages")) else {
        return;
    };
    let nvidia = sp.join("nvidia");
    let mut libs: Vec<PathBuf> = Vec::new();
    for pkg in glob_dir(&nvidia, "") {
        for l in glob_dir(&pkg.join("lib"), "lib") {
            if l.to_string_lossy().contains(".so") {
                libs.push(l);
            }
        }
    }
    if libs.is_empty() {
        return;
    }
    // Load in passes until nothing more succeeds (dependency order is unknown).
    let mut loaded = 0;
    loop {
        let before = libs.len();
        libs.retain(|l| unsafe {
            match libloading::os::unix::Library::open(Some(l), libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL) {
                Ok(lib) => {
                    std::mem::forget(lib);
                    loaded += 1;
                    false
                }
                Err(_) => true,
            }
        });
        if libs.is_empty() || libs.len() == before {
            break;
        }
    }
    log::debug!("preloaded {loaded} CUDA libraries from {}", nvidia.display());
}

pub fn runtime(cfg: &Gpu, data_dir: &Path) -> Result<&'static Runtime> {
    let r = RUNTIME.get_or_init(|| init(cfg, data_dir).map_err(|e| format!("{e:#}")));
    r.as_ref().map_err(|e| anyhow!("{e}"))
}

fn init(cfg: &Gpu, data_dir: &Path) -> Result<Runtime> {
    let lib = find_library(cfg, data_dir).ok_or_else(|| {
        anyhow!(
            "ONNX Runtime not found. Install it (Arch/CachyOS: `pacman -S onnxruntime-cuda` or `onnxruntime-cpu`, \
             elsewhere `pip install onnxruntime-gpu`) or set gpu.onnxruntime_lib / ORT_DYLIB_PATH"
        )
    })?;
    let wants_gpu = matches!(cfg.provider.as_str(), "auto" | "cuda" | "tensorrt");
    if wants_gpu {
        preload_cuda_wheels(&lib);
    }
    let env = ort::init_from(&lib)
        .with_context(|| format!("cannot load {}", lib.display()))?
        .with_name("tapdream")
        .build()
        .map_err(|e| anyhow!("cannot initialise ONNX Runtime: {e}"))?;
    let has_cuda = lib.parent().map(|d| d.join("libonnxruntime_providers_cuda.so").exists()).unwrap_or(false);
    let provider = match cfg.provider.as_str() {
        "cpu" => "cpu",
        "cuda" => "cuda",
        "tensorrt" => "tensorrt",
        _ if has_cuda => "cuda",
        _ => "cpu",
    }
    .to_string();
    Ok(Runtime {
        env,
        lib,
        provider,
        auto: !matches!(cfg.provider.as_str(), "cuda" | "tensorrt"),
        device_id: cfg.device_id,
        cpu_threads: cfg.cpu_threads,
        cache_dir: data_dir.join("cache").join("tensorrt"),
    })
}

impl Runtime {
    /// Loads a model on the configured device. With `provider = "auto"` it
    /// falls back to the CPU when the GPU cannot be used.
    pub fn session(&self, model: &Path) -> Result<(Session, String)> {
        if !model.exists() {
            bail!("model {} not found (run `tapdream models fetch`)", model.display());
        }
        let build = |provider: &str| -> Result<Session> {
            let mut b = Session::builder(&self.env)
                .map_err(|e| anyhow!("{e}"))?
                .with_optimization_level(GraphOptimizationLevel::All)
                .map_err(|e| anyhow!("{e}"))?;
            if self.cpu_threads > 0 {
                b = b.with_intra_threads(self.cpu_threads).map_err(|e| anyhow!("{e}"))?;
            }
            match provider {
                "cuda" => {
                    b = b
                        .with_execution_providers([ep::CUDA::default().with_device_id(self.device_id).build().error_on_failure()])
                        .map_err(|e| anyhow!("{e}"))?;
                }
                "tensorrt" => {
                    let _ = std::fs::create_dir_all(&self.cache_dir);
                    b = b
                        .with_execution_providers([
                            ep::TensorRT::default()
                                .with_device_id(self.device_id)
                                .with_fp16(true)
                                .with_engine_cache(true)
                                .with_engine_cache_path(self.cache_dir.display())
                                .build()
                                .error_on_failure(),
                            ep::CUDA::default().with_device_id(self.device_id).build(),
                        ])
                        .map_err(|e| anyhow!("{e}"))?;
                }
                _ => {}
            }
            b.commit_from_file(model).map_err(|e| anyhow!("cannot load {}: {e}", model.display()))
        };
        match build(&self.provider) {
            Ok(s) => Ok((s, self.provider.clone())),
            Err(e) if self.provider != "cpu" && self.auto => {
                log::warn!("GPU inference unavailable ({e:#}); using the CPU for {}", model.display());
                Ok((build("cpu")?, "cpu".into()))
            }
            Err(e) => Err(e),
        }
    }
}
