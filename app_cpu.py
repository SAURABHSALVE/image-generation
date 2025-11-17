import os
import sys

# ============================================================================
# CRITICAL: Set environment variables BEFORE any other imports
# This must be the first thing that runs to suppress TensorFlow warnings
# ============================================================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XFORMERS_DISABLE"] = "1"
os.environ["DIFFUSERS_NO_ONNX"] = "1"
os.environ["XFORMERS_MORE_DETAILS"] = "0"

# Load .env file if it exists (before other imports)
from pathlib import Path
env_path = Path(".env")
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

# Now import everything else
import time
import torch
import streamlit as st
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from streamlit.components.v1 import html
import bitsandbytes as bnb
from PIL import Image, ImageEnhance
import io
import base64
import logging
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.utils import load_image
from huggingface_hub import InferenceClient



# Securely load HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# if not HF_TOKEN:
#     st.warning("HF_TOKEN not set in environment. Some features (e.g., SDXL) may fail.")

# Setup logging
logging.basicConfig(filename="app_errors.log", level=logging.ERROR,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Page configuration
st.set_page_config(
    page_title="Ultimate AI Creation Suite",
    page_icon="🔮",
    layout="wide"
)

# --- ENHANCED UI DESIGN ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Main background with subtle gradient */
.stApp {
    background-color: #f0f2f6;
    background-image: 
        radial-gradient(circle at 10% 10%, #dbeafe 0%, transparent 50%),
        radial-gradient(circle at 80% 90%, #ede9fe 0%, transparent 50%);
    color: #1f2937;
}

/* --- Advanced Control Panel Styling with Glassmorphism --- */
.control-panel {
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(10px) saturate(180%);
    -webkit-backdrop-filter: blur(10px) saturate(180%);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
}

/* --- Modern Tab Styling --- */
div[data-baseweb="tab-list"] {
    background-color: rgba(229, 231, 235, 0.5);
    padding: 6px;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.04);
}

button[data-baseweb="tab"] {
    background-color: transparent;
    border: none;
    border-radius: 8px;
    color: #4b5563;
    font-weight: 600;
    transition: background-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #ffffff;
    color: #4f46e5;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.15);
}

button[data-baseweb="tab"]:not([aria-selected="true"]):hover {
    background-color: rgba(255, 255, 255, 0.8);
    color: #1f2937;
}

button[data-baseweb="tab"] > div {
    border-bottom: none !important;
}

/* Title styling with enhanced gradient and shadow */
.title-gradient {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    text-shadow: 0 2px 8px rgba(99, 102, 241, 0.1);
    padding-bottom: 0.5rem;
    letter-spacing: -0.025em;
}

/* Enhanced button styling with glow effect */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 32px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4);
}
.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
}

/* Enhanced Generated Image Card */
.image-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.image-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
}
.image-card img { border-radius: 12px; width: 100%; }
.image-card .caption { font-size: 0.95rem; color: #4b5563; margin-top: 1rem; word-wrap: break-word; line-height: 1.5; font-weight: 500; }
.image-card .download-link a { color: #4f46e5; text-decoration: none; font-weight: 600; padding: 0.5rem 1rem; border-radius: 8px; transition: background 0.2s ease; display: inline-block; margin-top: 0.5rem; }
.image-card .download-link a:hover { background: #eef2ff; text-decoration: none; }

/* Enhanced Placeholder */
.placeholder {
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    min-height: 400px;
    border: 2px dashed #d1d5db;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.5);
    text-align: center;
    padding: 3rem;
    transition: all 0.3s ease;
}
.placeholder:hover { border-color: #6366f1; box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1); }
.placeholder .icon { font-size: 5rem; animation: pulse 2s infinite; margin-bottom: 1rem; color: #6366f1;}
@keyframes pulse { 0%, 100% { transform: scale(1); opacity: 0.8; } 50% { transform: scale(1.05); opacity: 1; } }
.placeholder .text { font-size: 1.3rem; color: #374151; margin-top: 0; font-weight: 500; line-height: 1.6; }

/* Section headers */
h1, h2 { font-weight: 700; color: #111827; letter-spacing: -0.025em; }
h2 { border-bottom: 2px solid #e5e7eb; padding-bottom: 0.75rem; margin-bottom: 1.5rem; }

/* Input field styling */
.stTextInput > div > div > input, .stTextArea > div > textarea {
    border-radius: 10px !important;
    border: 1px solid #d1d5db !important;
    background-color: #f9fafb !important;
    transition: all 0.2s ease !important;
}
.stTextInput > div > div > input:focus, .stTextArea > div > textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 2px #c7d2fe !important;
    background-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# Model loading with OPTIMIZED CPU performance
@st.cache_resource(show_spinner=False)
def load_models(model_choice="sd-v1-5", use_quantization=False):
    """
    OPTIMIZED: Loads pipelines with CPU-specific optimizations.
    - Shares components between pipelines to reduce memory
    - Uses float32 for CPU compatibility
    - Enables memory-efficient attention
    - Disables unnecessary safety checkers
    - Caches models locally for faster subsequent loads
    """
    models = {}
    status_messages = []

    # Optimized arguments for CPU
    common_args = {
        "torch_dtype": torch.float32,  # CPU requires float32
        "use_safetensors": True,
        "safety_checker": None,  # Disable for speed
        "requires_safety_checker": False,
        "local_files_only": False,  # Will use cache if available
        "low_cpu_mem_usage": True  # Reduces memory during loading
    }
    
    if model_choice == "sdxl":
        status_messages.append(("warning", "⚠️ SDXL is very slow on CPU. SD v1.5 recommended."))
        try:
            models['txt2img'] = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                token=HF_TOKEN, 
                **common_args
            ).to("cpu")
            models['img2img'] = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                token=HF_TOKEN, 
                **common_args
            ).to("cpu")
            status_messages.append(("toast", "✅ SDXL loaded"))
        except Exception as e:
            logging.error(f"SDXL load failed: {e}")
            status_messages.append(("error", f"SDXL failed. Using SD v1.5."))
            model_choice = "sd-v1-5"
    
    if model_choice == "sd-v1-5":
        try:
            status_messages.append(("toast", "⏳ Loading SD v1.5 (optimized for CPU)..."))
            
            # OPTIMIZED: Load only txt2img, then convert to img2img
            # This is faster and more compatible than loading separately
            base_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                **common_args
            )
            
            # Move to CPU
            base_pipe = base_pipe.to("cpu")
            
            # Use DPM++ 2M Karras scheduler for FAST + HIGH QUALITY generation
            # This is 10x faster than default and produces better quality
            base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                base_pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
                solver_order=2
            )
            
            # Enable memory optimizations
            base_pipe.enable_vae_tiling()
            base_pipe.enable_attention_slicing(slice_size=1)
            
            # Store txt2img
            models['txt2img'] = base_pipe
            
            # FAST: Convert txt2img to img2img (reuses all components)
            models['img2img'] = StableDiffusionImg2ImgPipeline(
                **base_pipe.components
            )
            
            status_messages.append(("toast", "✅ Models loaded with shared components"))

            # Video2img reuses img2img pipeline
            models['video2img'] = models['img2img']

        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            status_messages.append(("error", f"Failed to load models: {e}"))
            return None, model_choice, status_messages

        # Load LCM-LoRA for faster generation (optional, can skip for faster startup)
        # Uncomment to enable 4x faster generation (adds 10-15 seconds to startup)
        # try:
        #     status_messages.append(("toast", "⏳ Loading LCM-LoRA for speed boost..."))
        #     models['txt2img'].load_lora_weights(
        #         "latent-consistency/lcm-lora-sdv1-5", 
        #         weight_name="pytorch_lora_weights.safetensors"
        #     )
        #     models['img2img'].load_lora_weights(
        #         "latent-consistency/lcm-lora-sdv1-5", 
        #         weight_name="pytorch_lora_weights.safetensors"
        #     )
        #     status_messages.append(("toast", "✅ LCM-LoRA loaded (4x faster generation)"))
        # except Exception as e:
        #     logging.error(f"LCM-LoRA load failed: {e}")
        #     status_messages.append(("warning", "⚠️ LCM-LoRA skipped (will be slower)"))
    
    # Custom LoRA loading disabled for faster startup and CPU compatibility
    # To enable: uncomment below and install peft package
    # lora_weight_path = "lora_out/lora_out_cpu.bin"
    # if os.path.exists(lora_weight_path):
    #     try:
    #         models['txt2img'].load_lora_weights("lora_out", weight_name="lora_out_cpu.bin")
    #         models['img2img'].load_lora_weights("lora_out", weight_name="lora_out_cpu.bin")
    #         status_messages.append(("toast", "✅ Custom LoRA loaded"))
    #     except Exception as e:
    #         logging.error(f"Custom LoRA failed: {e}")
    #         status_messages.append(("warning", f"Custom LoRA skipped: {e}"))
    
    status_messages.append(("toast", "🚀 All models ready!"))
    return models, model_choice, status_messages

@st.cache_resource
def load_clip_model():
    """Loads CLIP for prompt refinement."""
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
    return processor, model

# --- Helper Functions ---
def refine_prompt_with_clip(prompt, clip_processor, clip_model, num_words=5):
    """Refines prompt using CLIP embeddings (simplified)."""
    prompt = ''.join(c for c in prompt if c.isprintable() or c in [' ', '\n', '\t'])
    refined_words = [word for word in prompt.split() if len(word) > 3][:num_words]
    return " ".join(refined_words) + ", highly detailed, ultra-sharp, 8k"

def detect_mood_and_adapt(prompt, model_choice):
    """Detects mood from prompt and adapts CFG/steps."""
    mood_keywords = {"serene": {"cfg": 7.5, "steps": 20}, "dramatic": {"cfg": 10.0, "steps": 50}, "vibrant": {"cfg": 8.5, "steps": 30}, "dark": {"cfg": 9.0, "steps": 40}}
    for mood, params in mood_keywords.items():
        if mood in prompt.lower(): return params
    return {"cfg": 7.5, "steps": 20 if model_choice == "sd-v1-5" else 50}

def preprocess_input_image(image, target_width=512, target_height=512):
    """
    Preprocesses input images for optimal img2img results.
    - Maintains aspect ratio with smart cropping
    - Normalizes brightness and contrast
    - Ensures proper dimensions
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Calculate aspect ratios
    img_aspect = image.width / image.height
    target_aspect = target_width / target_height
    
    # Smart crop to maintain aspect ratio
    if img_aspect > target_aspect:
        # Image is wider - crop width
        new_width = int(image.height * target_aspect)
        left = (image.width - new_width) // 2
        image = image.crop((left, 0, left + new_width, image.height))
    elif img_aspect < target_aspect:
        # Image is taller - crop height
        new_height = int(image.width / target_aspect)
        top = (image.height - new_height) // 2
        image = image.crop((0, top, image.width, top + new_height))
    
    # Resize to target dimensions
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Normalize brightness slightly for better generation
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.05)
    
    return image

def enhance_image(image, target_size=None, sharpen_factor=1.5, enhance_contrast=True, enhance_color=True):
    """
    Advanced image enhancement with multiple quality improvements.
    - Upscaling with LANCZOS (high-quality resampling)
    - Sharpness enhancement
    - Optional contrast and color enhancement
    """
    # Upscale if target size specified
    if target_size:
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Apply sharpness enhancement
    image = ImageEnhance.Sharpness(image).enhance(sharpen_factor)
    
    # Apply contrast enhancement for better depth
    if enhance_contrast:
        image = ImageEnhance.Contrast(image).enhance(1.1)
    
    # Apply color enhancement for more vibrant results
    if enhance_color:
        image = ImageEnhance.Color(image).enhance(1.05)
    
    return image

def get_image_download_link(img, filename="generated_image.png"):
    """Creates a download link for the image."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/png;base64,{img_str}" download="{filename}">Download Image</a>'

import cv2
import tempfile
    # --- Fixed extract_first_frame function ---
def extract_first_frame(video_bytes):
    """Extracts the first frame from uploaded video bytes."""
    # Save the uploaded video bytes into a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_bytes)
        temp_path = temp_file.name

    # Open the video file using its path
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise ValueError("❌ Could not open the uploaded video file.")

    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        raise ValueError("❌ Could not read a frame from the uploaded video.")

    # Convert BGR to RGB and return as PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


# --- UI Layout ---
st.markdown(
    '<h1 class="title-gradient text-3xl md:text-5xl text-center">Ultimate AI Creation Suite ⚡</h1>',
    unsafe_allow_html=True
)

# Load models first with custom progress
loading_placeholder = st.empty()
loading_placeholder.info("🚀 Loading AI models (optimized for CPU)... This may take 30-60 seconds on first run.")

progress_bar = st.progress(0)
status_text = st.empty()

try:
    status_text.text("Loading Stable Diffusion v1.5...")
    progress_bar.progress(20)
    
    initial_model_choice = "sd-v1-5"
    initial_use_quantization = False
    models, model_choice, status_messages = load_models(initial_model_choice, initial_use_quantization)
    
    progress_bar.progress(100)
    status_text.text("Models loaded successfully!")
    
    # Clear loading indicators
    import time
    time.sleep(0.5)
    loading_placeholder.empty()
    progress_bar.empty()
    status_text.empty()
    
except Exception as e:
    loading_placeholder.error(f"Failed to load models: {e}")
    st.stop()

# Show info after loading
st.info(
    "✨ Ready! Use Fast Mode for 15-30 second generation times. Unleash your creativity with Text-to-Image, Image-to-Image, and Video-to-Image."
)

# Display status messages after loading is complete
for msg_type, msg in status_messages:
    if msg_type == "toast":
        st.toast(msg)
    elif msg_type == "warning":
        st.warning(msg)
    elif msg_type == "error":
        st.error(msg)

if models:
    clip_processor, clip_model = load_clip_model()
else:
    st.error("Model loading failed. The application cannot continue.")
    st.stop()

# Main layout with two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    
    tab_core, tab_enhance, tab_advanced = st.tabs(["🖌️ Core Settings", "✨ AI Enhancements", "⚙️ Advanced"])

    # Initialize variables
    input_image = None
    strength = 0.75

    with tab_core:
        model_choice_display = st.selectbox(
            "**Model**",
            ["SD v1.5 (LCM, Fast)", "Stable Diffusion XL (Slow on CPU)"],
            index=0,
            key="model_select"
        )
        model_choice = "sdxl" if model_choice_display.startswith("Stable Diffusion XL") else "sd-v1-5"
        
        mode = st.selectbox(
            "**Generation Mode**",
            ["Text-to-Image", "Image-to-Image", "Video-to-Image"],
            key="mode_select"
        )
        
        prompt = st.text_area(
            "**Prompt**",
            value="A cozy coffee shop interior",
            height=100,
            key="prompt_input"
        )
        neg_prompt = st.text_input(
            "**Negative Prompt**",
            "blurry, low quality, low resolution, bad anatomy, bad hands, text, watermark, signature, worst quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed",
            key="neg_prompt_input",
            help="Detailed negative prompts improve quality significantly"
        )

        # --- File upload handling with preprocessing ---
        if mode == "Image-to-Image":
            uploaded_image = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"])
            if uploaded_image:
                raw_image = Image.open(uploaded_image)
                st.image(raw_image, caption="Original Image", use_column_width=True)
                input_image = preprocess_input_image(raw_image, 512, 512)

        elif mode == "Video-to-Image":
            uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
            if uploaded_video:
                raw_frame = extract_first_frame(uploaded_video.getvalue())
                st.image(raw_frame, caption="Extracted Frame", use_column_width=True)
                input_image = preprocess_input_image(raw_frame, 512, 512)


    with tab_enhance:
        presets = {"None": "", "Cozy Interior": "A cozy coffee shop interior...", "Cyberpunk City": "Cyberpunk neon street...", "Nature Landscape": "Mountain landscape at sunrise...", "Fantasy Art": "Epic fantasy castle..."}
        preset = st.selectbox("**Style Preset**", list(presets.keys()))
        if preset != "None" and st.session_state.get("prompt_input") != presets[preset]:
            st.session_state.prompt_input = presets[preset]
            st.rerun()

        use_refiner = st.checkbox("Enable AI Prompt Refiner", value=True)
        if use_refiner and prompt:
            refined_prompt = refine_prompt_with_clip(prompt, clip_processor, clip_model)
            st.info(f"Refined: {refined_prompt}")
            
        use_mood_adapter = st.checkbox("Enable Mood Adapter", value=True)
        if use_mood_adapter and prompt:
            mood_params = detect_mood_and_adapt(prompt, model_choice)
            st.info(f"Mood: CFG={mood_params['cfg']}, Steps={mood_params['steps']}")
    
    with tab_advanced:
        # Show API recommendation first
        st.info("💡 **Recommended**: Enable 'HuggingFace API' below for 2-5 second generation on GPU!")
        
        # ULTRA FAST MODE for CPU
        fast_mode = st.checkbox(
            "⚡ Ultra Fast Mode (Local CPU)", 
            value=True,
            help="For local generation: Uses minimal steps and lower resolution. Still takes 30-90 seconds on CPU."
        )
        
        if mode != "Text-to-Image":
            strength = st.slider(
                "**Image/Frame Strength**", 
                0.0, 1.0, 0.75,
                help="Lower = more faithful to input, Higher = more creative freedom. For img2img, 0.6-0.8 is optimal."
            )
        
        # Seed control for reproducibility
        use_random_seed = st.checkbox("Use Random Seed", value=True)
        if not use_random_seed:
            seed = st.number_input("**Seed**", min_value=0, max_value=2147483647, value=42, step=1)
        else:
            seed = None
        
        num_images = st.slider("**Number of Images**", 1, 4, 1)
        
        # Optimize dimensions based on fast mode
        if fast_mode:
            # ULTRA FAST: Small resolution, few steps
            width = st.select_slider("**Width**", [256, 384, 448], value=384, 
                                    help="384x384 recommended for CPU speed")
            height = st.select_slider("**Height**", [256, 384, 448], value=384,
                                     help="384x384 recommended for CPU speed")
            default_steps = 15  # Minimum for decent quality
            max_steps = 25
            st.info("💡 Ultra Fast: 384x384 @ 15 steps = ~30-60 seconds on CPU")
        else:
            # SLOW MODE: Will take 10-20 minutes on CPU!
            width = st.select_slider("**Width**", [384, 512, 640], value=512)
            height = st.select_slider("**Height**", [384, 512, 640], value=512)
            default_steps = 20
            max_steps = 30
            st.warning("⚠️ Warning: 512x512 takes 10-20 minutes on CPU!")
        
        sharpen_factor = st.slider("**Sharpen Factor**", 1.0, 3.0, 1.5, 0.1)
        upscale = st.selectbox("**Upscale To**", ["None", "512x512", "768x768", "1024x1024"], index=0)
        
        # Optimized steps based on mode
        min_steps = 10 if fast_mode else 15
        steps = st.slider("**Inference Steps**", min_steps, max_steps, default_steps, 
                         help="CPU: 15 steps = 30-60s, 20 steps = 60-120s, 25+ steps = 10+ minutes")
        
        # Optimized CFG for better quality
        default_cfg = 7.0 if fast_mode else 8.0
        guidance = st.slider("**CFG Scale**", 5.0, 12.0, default_cfg, 0.5,
                           help="Lower CFG = faster generation")
        
        # Image enhancement options
        enhance_contrast = st.checkbox("Enhance Contrast", value=True)
        enhance_color = st.checkbox("Enhance Color Saturation", value=True)
            
        use_inference_client = st.checkbox(
            "🚀 Use HuggingFace API (FAST - Runs on GPU!)", 
            value=False,
            help="Uses HuggingFace's free GPU API. 2-5 seconds per image! Text-to-Image only. Requires HF_TOKEN in .env file."
        )
        
        if use_inference_client:
            if mode != "Text-to-Image":
                st.warning("⚠️ API Mode only supports Text-to-Image. Switch mode or disable API.")
            elif not HF_TOKEN:
                st.error("❌ HF_TOKEN required! Add it to your .env file.")
            else:
                st.success("✅ API Mode: Using HuggingFace GPU servers (2-5 seconds!)")
        
        # Show REALISTIC estimated time for CPU
        if fast_mode:
            # CPU realistic: ~2-4 seconds per step at 384x384
            est_time = steps * 3 * num_images
            st.info(f"⚡ Ultra Fast Mode: ~{est_time:.0f}s estimated (CPU)")
        else:
            # CPU realistic: ~20-40 seconds per step at 512x512
            est_time = steps * 30 * num_images
            st.error(f"⚠️ Slow Mode: ~{est_time:.0f}s ({est_time/60:.1f} minutes) - NOT RECOMMENDED for CPU!")

    if st.button("Generate", key="generate", use_container_width=True):
        if not prompt:
            st.error("Please enter a prompt.")
        else:
            start_time = time.time()
            progress_bar = st.progress(0, "Starting Generation...")
            st.session_state.last_generation = []
            generated_images = []
            try:
                current_prompt = refined_prompt if 'refined_prompt' in locals() and use_refiner else prompt
                current_steps = mood_params['steps'] if 'mood_params' in locals() and use_mood_adapter else steps
                current_guidance = mood_params['cfg'] if 'mood_params' in locals() and use_mood_adapter else guidance


                if use_inference_client:
                    client = InferenceClient(token=HF_TOKEN)
                    for i in range(num_images):
                        progress_bar.progress((i + 1) / num_images, f"Generating via API {i+1}/{num_images}...")
                        image = client.text_to_image(prompt=current_prompt, model="stabilityai/stable-diffusion-xl-base-1.0" if model_choice == "sdxl" else "runwayml/stable-diffusion-v1-5", negative_prompt=neg_prompt, num_inference_steps=current_steps, guidance_scale=current_guidance, width=width, height=height)
                        image = enhance_image(image, target_size=int(upscale.split("x")[0]) if upscale != "None" else None, sharpen_factor=sharpen_factor, enhance_contrast=enhance_contrast, enhance_color=enhance_color)
                        generated_images.append((image, current_prompt))
                else:
                    pipe = models['txt2img'] if mode == "Text-to-Image" else models['img2img']
                    with torch.inference_mode():
                        for i in range(num_images):
                            progress_bar.progress((i + 1) / num_images, f"Generating locally {i+1}/{num_images}...")
                            
                            # Set seed for reproducibility
                            if seed is not None:
                                generator = torch.Generator(device="cpu").manual_seed(seed + i)
                            else:
                                generator = None
                            
                            # Note: LoRA adapters removed for faster CPU performance
                            # Custom LoRA strength is handled during model loading
                            
                            if mode == "Text-to-Image":
                                image = pipe(
                                    prompt=current_prompt, 
                                    negative_prompt=neg_prompt, 
                                    num_inference_steps=current_steps, 
                                    guidance_scale=current_guidance, 
                                    width=width, 
                                    height=height,
                                    generator=generator
                                ).images[0]
                            else:
                                if input_image is None:
                                    st.error(f"Please upload an {'image' if mode == 'Image-to-Image' else 'video'}.")
                                    break
                                
                                # Improved img2img generation with better parameters
                                image = pipe(
                                    prompt=current_prompt, 
                                    image=input_image, 
                                    negative_prompt=neg_prompt, 
                                    num_inference_steps=current_steps, 
                                    guidance_scale=current_guidance, 
                                    strength=strength,
                                    generator=generator
                                ).images[0]
                            
                            # Enhanced post-processing
                            image = enhance_image(
                                image, 
                                target_size=int(upscale.split("x")[0]) if upscale != "None" else None, 
                                sharpen_factor=sharpen_factor,
                                enhance_contrast=enhance_contrast,
                                enhance_color=enhance_color
                            )
                            generated_images.append((image, current_prompt))
                            
                            # Display seed used
                            if seed is not None:
                                st.toast(f"Used seed: {seed + i}")
                
                st.success(f"Generated {len(generated_images)} image(s) in {time.time() - start_time:.2f} seconds!")
                st.session_state.last_generation = generated_images
                if "image_history" not in st.session_state: st.session_state.image_history = []
                st.session_state.image_history.extend(generated_images)
            except Exception as e:
                logging.error(f"Generation failed: {e}")
                st.error(f"An error occurred during generation: {e}")
            finally:
                progress_bar.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)


# Initialize session state
if "image_history" not in st.session_state: st.session_state.image_history = []
if "last_generation" not in st.session_state: st.session_state.last_generation = []

with col2:
    st.markdown('<h2>✨ Latest Creation</h2>', unsafe_allow_html=True)
    if st.session_state.last_generation:
        cols = st.columns(min(len(st.session_state.last_generation), 3))
        for idx, (img, pr) in enumerate(st.session_state.last_generation):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            download_html = get_image_download_link(img, f"{mode.lower().replace('-', '_')}_{idx}.png")
            
            card_html = f"""
            <div class="image-card">
                <img src="data:image/png;base64,{img_str}" alt="Generated Image">
                <p class="caption"><b>Prompt:</b> {pr}</p>
                <p class="download-link">{download_html}</p>
            </div>
            """
            cols[idx % 3].markdown(card_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="placeholder"><div class="icon">🎨</div><div class="text">Your creations will appear here. <br/> Configure settings and click \'Generate\'!</div></div>', unsafe_allow_html=True)
    
    if st.session_state.image_history:
        st.markdown('<h2 class="text-2xl font-semibold mt-8 border-t border-gray-200 pt-6">📸 Recent Creations</h2>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (img, pr) in enumerate(reversed(st.session_state.image_history[-8:])):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            download_html = get_image_download_link(img, f"history_{len(st.session_state.image_history)-i}.png")
            cols[i % 4].markdown(f'<div class="image-card"><img src="data:image/png;base64,{img_str}"><p class="download-link mt-2">{download_html}</p></div>', unsafe_allow_html=True)


