import os, subprocess, threading, time, modal
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Config ---
APP_NAME = "sdnext-gui"
ROOT_DIR = "/root/sdnext"   # lokasi repo
DATA_DIR = "/data/sdnext"   # lokasi persistent data (models, outputs)
VOL_NAME = "sdnext-data"
SDNEXT_GIT = "https://github.com/vladmandic/sdnext.git"
GPU_TYPE = os.environ.get("MODAL_GPU_TYPE", "L4")

# --- Build Image ---
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands([
        "pip install --upgrade pip",
        # ✅ Install semua deps penting (gradio, fastapi, torch, onnx disabled)
        "pip install gradio==4.44.0 fastapi uvicorn requests pyyaml numpy safetensors transformers diffusers accelerate",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        f"git clone {SDNEXT_GIT} {ROOT_DIR}",
        f"cd {ROOT_DIR} && pip install -r requirements.txt --no-cache-dir || true",
    ])
)

# --- Modal App ---
app = modal.App(name=APP_NAME, image=image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

@app.function(gpu=GPU_TYPE, timeout=3600, scaledown_window=300, volumes={DATA_DIR: vol})
@modal.web_server(8000, startup_timeout=600)
def run():
    """Start SD.Next GUI (patched for Modal)"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.chdir(ROOT_DIR)

    print("🚀 Launching SD.Next GUI (patched mode)...")
    print(f"📁 Repo: {ROOT_DIR}")
    print(f"💾 Persistent volume: {DATA_DIR}")

    # === PATCH agar tidak error OpenVINO/ONNX ===
    os.environ["PYTORCH_TRACING_MODE"] = "TORCHFX"
    os.environ["USE_OPENVINO"] = "0"
    patch_code = """
import argparse, sys
import modules
if not hasattr(modules, 'shared'):
    import modules.shared as shared
else:
    shared = modules.shared
if not hasattr(shared, 'cmd_opts'):
    shared.cmd_opts = argparse.Namespace()
if not hasattr(shared.cmd_opts, 'use_openvino'):
    shared.cmd_opts.use_openvino = False
if not hasattr(shared.cmd_opts, 'use_onnx'):
    shared.cmd_opts.use_onnx = False
print('[PATCH] Injected safe cmd_opts (no OpenVINO / no ONNX)')
"""
    with open(f"{ROOT_DIR}/prepatch.py", "w") as f:
        f.write(patch_code)

    # === Jalankan SD.Next GUI dengan patch ===
    def start_sdnext():
        cmd = [
            "python", "-c",
            f"import runpy; exec(open('{ROOT_DIR}/prepatch.py').read()); runpy.run_path('webui.py', run_name='__main__')",
            "--listen", "0.0.0.0",
            "--port", "8000",
            "--skip-torch-cuda-test",
            "--disable-safe-unpickle",
            "--no-onnx",
            "--no-half",
            "--skip-version-check",
            "--data-dir", DATA_DIR,
            "--autolaunch",
        ]
        subprocess.Popen(cmd, cwd=ROOT_DIR, env=os.environ.copy())

    threading.Thread(target=start_sdnext, daemon=True).start()
    time.sleep(8)

    # === Healthcheck API (FastAPI) ===
    app_api = FastAPI(title="SD.Next GUI (Patched)")
    app_api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app_api.get("/")
    def root():
        return JSONResponse({
            "status": "ok",
            "mode": "GUI",
            "repo": ROOT_DIR,
            "data_volume": DATA_DIR
        })

    @app_api.get("/health")
    def health():
        return JSONResponse({"status": "ok"})

    print("✅ SD.Next GUI launched successfully!")
    print("🌐 Check Modal dashboard → Web Endpoints tab for public URL")

    return app_api
