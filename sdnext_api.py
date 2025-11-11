# app_modal_sdnext.py - Backend SD.Next dari repo vladmandic
import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
import torch
import os
import sys
import time

# Konfigurasi Modal
app = modal.App("sdnext-backend-original")
image = modal.Image.debian_slim().apt_install(
    "git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1"
).pip_install(
    "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"
).pip_install(
    "diffusers", "transformers", "accelerate", "safetensors", "einops",
    "opencv-python", "Pillow", "fastapi", "uvicorn", "pydantic", "tqdm",
    "k-diffusion", "gradio", "psutil", "requests", "numpy", "scipy"
)

# Pydantic models
class Text2ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    sampler: str = "Euler"
    model: str = "stable-diffusion-v1-5"

class ImageResponse(BaseModel):
    image_base64: str
    info: dict

@app.cls(
    gpu="A100",  # Rekomendasi untuk SD.Next
    timeout=600,  # 10 menit untuk load model
    container_idle_timeout=300,
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")]  # Jika perlu HF token
)
class SDNextOriginal:
    def __enter__(self):
        print("🚀 Initializing SD.Next from vladmandic repo...")
        
        # Clone repo jika belum ada
        if not os.path.exists("/sdnext"):
            os.system("git clone https://github.com/vladmandic/sdnext.git /sdnext")
        
        sys.path.append("/sdnext")
        os.chdir("/sdnext")
        
        # Install requirements
        os.system("pip install -r requirements.txt --no-deps --quiet")
        os.system("pip install -r requirements-extra.txt --no-deps --quiet")
        
        # Import SD.Next modules
        try:
            from modules import paths, shared, devices, sd_models, sd_samplers
            from modules.processing import StableDiffusionProcessingTxt2Img, process_images
            from modules.shared import opts
            
            # Initialize
            shared.cmd_opts = type('obj', (object,), {
                'ckpt': None,
                'data_dir': '/sdnext',
                'models_dir': '/models',
                'no_download': True,
                'skip_install': True
            })()
            
            paths.script_path = "/sdnext"
            shared.models_path = "/models"
            
            # Buat folder models
            os.makedirs("/models/Stable-diffusion", exist_ok=True)
            os.makedirs("/models/VAE", exist_ok=True)
            os.makedirs("/models/Lora", exist_ok=True)
            
            print("✅ SD.Next modules imported")
            
            # Load model pertama kali (cache)
            self._load_model("stable-diffusion-v1-5")
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading SD.Next: {e}")
            raise e
    
    def _load_model(self, model_name="stable-diffusion-v1-5"):
        """Load model ke GPU"""
        from modules import shared, sd_models
        
        if model_name == "stable-diffusion-xl-base-1.0":
            model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            model_path = "runwayml/stable-diffusion-v1-5"
        
        # Download model jika belum ada
        if not os.path.exists(f"/models/Stable-diffusion/{model_name}"):
            print(f"📥 Downloading model: {model_path}")
            os.system(f"huggingface-cli download {model_path} --local-dir /models/Stable-diffusion/{model_name}")
        
        # Load checkpoint
        shared.opts.sd_model_checkpoint = f"/models/Stable-diffusion/{model_name}"
        sd_models.load_model()
        
        self.current_model = model_name
        print(f"✅ Model {model_name} loaded")
    
    @modal.method()
    def generate(self, request: Text2ImageRequest):
        try:
            from modules import shared, devices, sd_models, sd_samplers
            from modules.processing import StableDiffusionProcessingTxt2Img, process_images
            
            # Switch model jika berbeda
            if not hasattr(self, 'current_model') or self.current_model != request.model:
                self._load_model(request.model)
            
            # Setup processing
            devices.torch_gc()  # Clear cache
            
            p = StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                cfg_scale=request.cfg_scale,
                sampler_name=request.sampler,
                seed=request.seed if request.seed != -1 else -1,
                batch_size=1,
            )
            
            # Generate
            start_time = time.time()
            processed = process_images(p)
            generation_time = time.time() - start_time
            
            # Convert to base64
            img = processed.images[0]
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Info
            info = {
                "model": request.model,
                "seed": processed.seed,
                "steps": request.steps,
                "cfg_scale": request.cfg_scale,
                "sampler": request.sampler,
                "generation_time": f"{generation_time:.2f}s",
                "size": f"{request.width}x{request.height}"
            }
            
            return ImageResponse(image_base64=img_base64, info=info)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

# FastAPI app
web_app = FastAPI(title="SD.Next API")

@web_app.post("/generate", response_model=ImageResponse)
async def generate_image(request: Text2ImageRequest):
    """Generate image using SD.Next"""
    sd = SDNextOriginal()
    return sd.generate.remote(request)

@web_app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "sdnext-original-backend",
        "gpu": "A100",
        "repo": "vladmandic/sdnext"
    }

@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app

# CLI untuk testing lokal
@app.local_entrypoint()
def main():
    print("Testing SD.Next backend...")
    request = Text2ImageRequest(
        prompt="a cat wearing a wizard hat, photorealistic",
        steps=10,
        model="stable-diffusion-v1-5"
    )
    
    sd = SDNextOriginal()
    response = sd.generate.remote(request)
    print(f"✅ Generated! Info: {response.info}")
