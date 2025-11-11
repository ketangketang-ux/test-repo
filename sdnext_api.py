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
        # deps utama
        "pip install gradio==4.44.0 fastapi uvicorn requests pyyaml numpy safetensors transformers diffusers accelerate",
        # torch wheel (sesuaikan kalau Modal ganti CUDA)
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        # clone repo
        f"git clone {SDNEXT_GIT} {ROOT_DIR} || true",
        # try install requirements (tolerant)
        f"cd {ROOT_DIR} && pip install -r requirements.txt --no-cache-dir || true",
    ])
)

# === MODAL APP / VOLUME ===
app = modal.App(name=APP_NAME, image=image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


def _patch_shared_file(repo_root):
    """
    Ensure modules/shared.py has safe defaults for cmd_opts.use_openvino / use_onnx.
    This prepends a small patch header if not already patched.
    """
    shared_path = os.path.join(repo_root, "modules", "shared.py")
    try:
        if not os.path.exists(shared_path):
            # nothing to patch (repo might be incomplete), just return
            return False
        with open(shared_path, "r", encoding="utf-8") as f:
            content = f.read()
        marker = "# PATCHED_BY_MODAL_SAFE_CMD_OPTS"
        if marker in content:
            return True  # already patched
        header = f'''# {marker}
# Added by modal deploy to ensure cmd_opts defaults exist (avoid AttributeError on use_openvino/use_onnx)
try:
    import argparse
    # ensure cmd_opts exists in this module namespace
    if 'cmd_opts' not in globals():
        cmd_opts = argparse.Namespace()
    if not hasattr(cmd_opts, 'use_openvino'):
        cmd_opts.use_openvino = False
    if not hasattr(cmd_opts, 'use_onnx'):
        cmd_opts.use_onnx = False
except Exception:
    pass

'''
        new_content = header + content
        with open(shared_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    except Exception as e:
        print("! Warning: failed to patch modules/shared.py:", e)
        return False


@app.function(
    gpu=GPU_TYPE,
    timeout=3600,
    scaledown_window=300,
    volumes={DATA_DIR: vol},
)
@modal.web_server(8000, startup_timeout=600)
def run():
    """Start SD.Next GUI with safe patching for shared.cmd_opts"""
    os.makedirs(DATA_DIR, exist_ok=True)

    # ensure repo exists in container (image build does clone; keep check)
    if not os.path.exists(ROOT_DIR):
        try:
            subprocess.run(f"git clone {SDNEXT_GIT} {ROOT_DIR}", shell=True, check=True)
        except Exception as e:
            print("Error cloning repo at runtime:", e)

    # safe-patch shared.py so AttributeError use_openvino won't happen
    patched = _patch_shared_file(ROOT_DIR)
    if patched:
        print("✅ modules/shared.py patched for safe cmd_opts defaults.")
    else:
        print("⚠️ modules/shared.py not patched (file missing or patch failed).")

    # cd into repo
    os.chdir(ROOT_DIR)
    print("📁 Current repo path:", ROOT_DIR)
    print("💾 Data volume path:", DATA_DIR)

    # start SD.Next GUI (we avoid ONNX/OpenVINO features to reduce incompat)
    def start_sdnext():
        cmd = [
            "python", "webui.py",
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
        # run non-blocking
        subprocess.Popen(cmd, cwd=ROOT_DIR, env=os.environ.copy())

    t = threading.Thread(target=start_sdnext, daemon=True)
    t.start()

    # give it a few seconds to attempt startup
    time.sleep(8)

    # provide a simple FastAPI wrapper for health/status
    app_api = FastAPI(title="SD.Next GUI (Modal patched)")
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
            "message": "SD.Next GUI launching (patched)",
            "repo": ROOT_DIR,
            "data_volume": DATA_DIR
        })

    @app_api.get("/health")
    def health():
        return JSONResponse({"status": "ok"})

    print("🚀 SD.Next launch attempted. Check Modal dashboard -> Logs / Web Endpoints for URL.")
    return app_api
