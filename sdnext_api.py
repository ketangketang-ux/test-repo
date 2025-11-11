import os
import subprocess
import threading
import time
import modal
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# === CONFIG ===
APP_NAME = "sdnext-gui"
ROOT_DIR = "/root/sdnext"
DATA_DIR = "/data/sdnext"
VOL_NAME = "sdnext-data"
SDNEXT_GIT = "https://github.com/vladmandic/sdnext.git"
GPU_TYPE = os.environ.get("MODAL_GPU_TYPE", "L4")

# === IMAGE BUILD ===
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands([
        "pip install --upgrade pip",
        "pip install gradio==4.44.0 fastapi uvicorn requests pyyaml numpy safetensors transformers diffusers accelerate",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        f"git clone {SDNEXT_GIT} {ROOT_DIR}",
        f"cd {ROOT_DIR} && pip install -r requirements.txt --no-cache-dir || true",
    ])
)

# === MODAL CONFIG ===
app = modal.App(name=APP_NAME, image=image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


@app.function(
    gpu=GPU_TYPE,
    timeout=3600,
    scaledown_window=300,
    volumes={DATA_DIR: vol},
)
@modal.web_server(8000, startup_timeout=600)
def run():
    """Start SD.Next GUI backend (patched version)"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.chdir(ROOT_DIR)

    print("🚀 Launching SD.Next GUI (no ONNX patch)...")
    print(f"📁 Repo: {ROOT_DIR}")
    print(f"💾 Persistent volume: {DATA_DIR}")

    # === PRE-PATCH ===
    patch_code = """
import sys, types
fake_mod = types.ModuleType('modules.onnx_impl')
fake_mod.ort = None
fake_mod.execution_providers = []
fake_mod.DynamicSessionOptions = type('Dummy', (), {})()
sys.modules['modules.onnx_impl'] = fake_mod
print('[PATCH] Disabled ONNX module safely.')
"""
    with open(f"{ROOT_DIR}/disable_onnx.py", "w") as f:
        f.write(patch_code)

    # === START SD.NEXT ===
    def start_sdnext():
        cmd = [
            "python",
            "-c",
            f"import runpy; exec(open('{ROOT_DIR}/disable_onnx.py').read()); runpy.run_path('webui.py', run_name='__main__')",
            "--listen", "0.0.0.0",
            "--port", "8000",
            "--skip-torch-cuda-test",
            "--disable-safe-unpickle",
            "--no-half",
            "--skip-version-check",
            "--data-dir", DATA_DIR,
            "--autolaunch",
        ]
        subprocess.Popen(cmd, cwd=ROOT_DIR, env=os.environ.copy())

    thread = threading.Thread(target=start_sdnext)
    thread.daemon = True
    thread.start()
    time.sleep(8)

    # === HEALTH API ===
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
    print("🌐 Check your Modal dashboard → Web Endpoints tab for public URL")

    return app_api
