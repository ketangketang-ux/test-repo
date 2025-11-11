# CELL 1-4: Sama seperti sebelumnya (setup, config, download, token)

# CELL 5: DEPLOY (gunakan script di atas di GitHub, atau direct deploy)
# CELL 6-7: Parse URL & Test

# CELL 8: Gradio UI FULL
import gradio as gr
import requests
import base64
import io

# Daftar model
AVAILABLE_MODELS = [
    "stable-diffusion-v1-5", "stable-diffusion-xl-base-1.0",
    "anything-v5", "dreamshaper", "gpt"
]

# Daftar sampler
SAMPLERS = ["Euler", "Euler a", "LMS", "Heun", "DPM2", "DPM2 a", "DPM++ 2S a", "DPM++ 2M", "DPM++ SDE", "DDIM", "PLMS", "UniPC"]

def get_models():
    """Get list of models from Modal API"""
    try:
        response = requests.get(f"{MODAL_URL}/models", timeout=10)
        if response.status_code == 200:
            return response.json()['models']
    except:
        pass
    return AVAILABLE_MODELS

def generate_txt2img(prompt, negative, width, height, steps, cfg, seed, sampler, model):
    """Generate image from text"""
    try:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg,
            "seed": seed,
            "sampler": sampler,
            "model": model
        }
        
        response = requests.post(f"{MODAL_URL}/txt2img", json=payload, timeout=600)
        
        if response.status_code == 200:
            result = response.json()
            img_data = base64.b64decode(result["image_base64"])
            img = Image.open(io.BytesIO(img_data))
            info = json.dumps(result["info"], indent=2)
            return img, f"✅ Success!\n\nInfo:\n{info}"
        else:
            return None, f"❌ Error: {response.status_code}\n{response.text}"
            
    except Exception as e:
        return None, f"❌ Exception: {str(e)}"

def generate_img2img(prompt, negative, image, strength, steps, cfg, seed, sampler, model):
    """Generate image from image"""
    try:
        # Convert PIL image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative,
            "image_base64": img_base64,
            "strength": strength,
            "steps": steps,
            "cfg_scale": cfg,
            "seed": seed,
            "sampler": sampler,
            "model": model
        }
        
        response = requests.post(f"{MODAL_URL}/img2img", json=payload, timeout=600)
        
        if response.status_code == 200:
            result = response.json()
            img_data = base64.b64decode(result["image_base64"])
            img = Image.open(io.BytesIO(img_data))
            info = json.dumps(result["info"], indent=2)
            return img, f"✅ Success!\n\nInfo:\n{info}"
        else:
            return None, f"❌ Error: {response.status_code}\n{response.text}"
            
    except Exception as e:
        return None, f"❌ Exception: {str(e)}"

def open_webui():
    """Open Modal WebUI in new tab"""
    import webbrowser
    webbrowser.open(f"{MODAL_URL}?__theme=dark")
    return f"✅ Opening {MODAL_URL} in new tab!"

# Create UI
with gr.Blocks(title="SD.Next Full - Modal GPU", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 SD.Next Full Version (vladmandic) + Modal GPU")
    gr.Markdown(f"**Backend**: `{MODAL_URL}`")
    
    with gr.Tab("Text-to-Image"):
        with gr.Row():
            with gr.Column():
                txt_prompt = gr.Textbox(label="📝 Prompt", lines=3, placeholder="masterpiece, best quality, ...")
                txt_negative = gr.Textbox(label="⚠️ Negative Prompt", lines=2, value="nsfw, lowres, bad anatomy, text, error, blurry")
                
                with gr.Row():
                    txt_model = gr.Dropdown(AVAILABLE_MODELS, value="stable-diffusion-v1-5", label="🏗️ Model")
                    txt_sampler = gr.Dropdown(SAMPLERS, value="Euler a", label="🎨 Sampler")
                
                with gr.Row():
                    txt_width = gr.Slider(256, 1024, 512, step=64, label="📐 Width")
                    txt_height = gr.Slider(256, 1024, 512, step=64, label="📐 Height")
                
                with gr.Row():
                    txt_steps = gr.Slider(10, 50, 25, step=1, label="🔄 Steps")
                    txt_cfg = gr.Slider(1, 15, 7.0, step=0.5, label="🔧 CFG Scale")
                
                txt_seed = gr.Number(-1, label="🎲 Seed (-1 = random)")
                txt_btn = gr.Button("🎨 Generate", variant="primary", size="lg")
                
            with gr.Column():
                txt_output = gr.Image(label="🖼️ Output", height=512)
                txt_info = gr.Textbox(label="ℹ️ Info", lines=6)
    
    with gr.Tab("Image-to-Image"):
        with gr.Row():
            with gr.Column():
                img_prompt = gr.Textbox(label="📝 Prompt", lines=3)
                img_negative = gr.Textbox(label="⚠️ Negative Prompt", lines=2)
                img_input = gr.Image(label="📷 Input Image", type="pil")
                
                img_strength = gr.Slider(0.0, 1.0, 0.75, step=0.05, label="💪 Denoising Strength")
                
                with gr.Row():
                    img_model = gr.Dropdown(AVAILABLE_MODELS, value="stable-diffusion-v1-5", label="Model")
                    img_sampler = gr.Dropdown(SAMPLERS, value="Euler a", label="Sampler")
                
                with gr.Row():
                    img_steps = gr.Slider(10, 50, 25, step=1, label="Steps")
                    img_cfg = gr.Slider(1, 15, 7.0, step=0.5, label="CFG Scale")
                
                img_seed = gr.Number(-1, label="Seed")
                img_btn = gr.Button("🔄 Generate img2img", variant="primary")
                
            with gr.Column():
                img_output = gr.Image(label="🖼️ Output", height=512)
                img_info = gr.Textbox(label="ℹ️ Info", lines=6)
    
    with gr.Tab("🎛️ WebUI"):
        gr.Markdown("### Buka UI Bawaan SD.Next di Modal")
        gr.Markdown("Fitur: ControlNet, Upscale, Settings Lengkap")
        webui_btn = gr.Button("🚀 Launch SD.Next WebUI", variant="secondary")
        webui_output = gr.Textbox(label="Status", lines=2)
    
    # Event handlers
    txt_btn.click(
        fn=generate_txt2img,
        inputs=[txt_prompt, txt_negative, txt_width, txt_height, txt_steps, txt_cfg, txt_seed, txt_sampler, txt_model],
        outputs=[txt_output, txt_info]
    )
    
    img_btn.click(
        fn=generate_img2img,
        inputs=[img_prompt, img_negative, img_input, img_strength, img_steps, img_cfg, img_seed, img_sampler, img_model],
        outputs=[img_output, img_info]
    )
    
    webui_btn.click(fn=open_webui, outputs=webui_output)

# CELL 9: Launch
demo.launch(share=True, debug=True, height=800)
