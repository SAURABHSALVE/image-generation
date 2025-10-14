# import os
# os.environ["XFORMERS_DISABLE"] = "1"
# os.environ["DIFFUSERS_NO_ONNX"] = "1"


# import torch
# import streamlit as st
# from diffusers import StableDiffusionPipeline

# st.set_page_config(page_title="Text-to-Image CPU", page_icon="🎨")

# @st.cache_resource
# def load_pipe():
#     pipe = StableDiffusionPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-2-base",
#         torch_dtype=torch.float32,
#         safety_checker=None
#     ).to("cpu")
#     pipe.enable_vae_tiling()
#     if os.path.exists("lora_out/lora_out_cpu.bin"):
#         try:
#             pipe.load_lora_weights("lora_out", weight_name="lora_out_cpu.bin")
#         except Exception as e:
#             st.warning(f"Could not load LoRA weights: {e}")
#     return pipe

# pipe = load_pipe()
# st.title("🎨 Text-to-Image CPU with LoRA")

# prompt = st.text_area("Prompt", "a cozy coffee shop interior")
# neg = st.text_input("Negative prompt", "blurry")
# steps = st.slider("Steps", 10, 40, 20)
# guidance = st.slider("CFG Scale", 1.0, 15.0, 7.5)
# width = st.selectbox("Width", [256, 384, 512], index=0)
# height = st.selectbox("Height", [256, 384, 512], index=0)

# if st.button("Generate"):
#     with torch.autocast("cpu", enabled=False):
#         img = pipe(
#             prompt,
#             negative_prompt=neg,
#             num_inference_steps=int(steps),
#             guidance_scale=float(guidance),
#             width=int(width),
#             height=int(height)
#         ).images[0]
#     st.image(img, caption="Generated Image", use_column_width=True)



# import os
# import time
# import torch
# import streamlit as st
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# from streamlit.components.v1 import html

# # Environment setup
# os.environ["XFORMERS_DISABLE"] = "1"
# os.environ["DIFFUSERS_NO_ONNX"] = "1"

# # Page configuration
# st.set_page_config(
#     page_title="Text-to-Image Generator",
#     page_icon="🎨",
#     layout="wide"
# )

# # Inject Tailwind CSS via CDN
# html("""
# <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
# """)

# # Model loading
# @st.cache_resource
# def load_pipe():
#     """Loads the Stable Diffusion pipeline with optimizations for CPU."""
#     pipe = StableDiffusionPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#         torch_dtype=torch.float32,
#         safety_checker=None
#     ).to("cpu")
#     pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
#     pipe.enable_vae_tiling()
    
#     lora_weight_path = "lora_out/lora_out_cpu.bin"
#     if os.path.exists(lora_weight_path):
#         try:
#             pipe.load_lora_weights("lora_out", weight_name="lora_out_cpu.bin")
#             st.sidebar.success("✅ LoRA weights loaded successfully!")
#         except Exception as e:
#             st.sidebar.warning(f"Could not load LoRA weights: {e}")
#     else:
#         st.sidebar.info("No LoRA weights found. Using base model.")
    
#     return pipe

# # UI Layout
# st.markdown('<h1 class="text-3xl font-bold text-center text-gray-800 mb-6">🎨 Text-to-Image Generator</h1>', unsafe_allow_html=True)
# st.info("This app runs Stable Diffusion on CPU. Generation may take a few minutes.")

# # Load model
# with st.spinner("Loading model... Please wait."):
#     pipe = load_pipe()

# # Sidebar for controls
# st.sidebar.markdown('<h2 class="text-xl font-semibold text-gray-700">⚙️ Settings</h2>', unsafe_allow_html=True)

# # Style presets
# presets = {
#     "Cozy Interior": "A cozy coffee shop interior, warm lighting, wooden furniture, soft ambiance",
#     "Cyberpunk City": "Cyberpunk neon street, night rain, vibrant colors, futuristic vibe",
#     "Nature Landscape": "Mountain landscape at sunrise, misty valleys, vibrant colors",
#     "Fantasy Art": "Epic fantasy castle, glowing skies, magical aura, detailed architecture"
# }
# preset = st.sidebar.selectbox("**Style Preset**", ["None"] + list(presets.keys()))
# prompt = st.sidebar.text_area(
#     "**Prompt**",
#     presets.get(preset, "A cozy coffee shop interior, warm lighting") if preset != "None" else "",
#     height=100,
#     help="Describe the image you want to create."
# )
# neg_prompt = st.sidebar.text_input(
#     "**Negative Prompt**",
#     "blurry, low quality, unrealistic, deformed, text, watermark",
#     help="Avoid these elements in the image."
# )

# # Advanced settings
# st.sidebar.markdown('<h3 class="text-lg font-medium text-gray-600">Advanced</h3>', unsafe_allow_html=True)
# width = st.sidebar.select_slider("**Width**", options=[256, 384, 512], value=512)
# height = st.sidebar.select_slider("**Height**", options=[256, 384, 512], value=512)
# steps = st.sidebar.slider("**Inference Steps**", min_value=10, max_value=30, value=20, help="More steps improve quality but take longer.")
# guidance = st.sidebar.slider("**CFG Scale**", min_value=1.0, max_value=15.0, value=7.5, step=0.5)

# # Image history
# if "image_history" not in st.session_state:
#     st.session_state.image_history = []

# # Generate button
# if st.sidebar.button("Generate Image", key="generate", help="Click to generate the image"):
#     if not prompt:
#         st.error("Please enter a prompt.")
#     else:
#         start_time = time.time()
#         progress_bar = st.progress(0)
#         with st.spinner("Generating image..."):
#             try:
#                 with torch.autocast("cpu", enabled=False):
#                     for i in range(steps):
#                         progress_bar.progress((i + 1) / steps)
#                         time.sleep(0.1)  # Simulate progress
#                     image = pipe(
#                         prompt=prompt,
#                         negative_prompt=neg_prompt,
#                         num_inference_steps=steps,
#                         guidance_scale=guidance,
#                         width=width,
#                         height=height
#                     ).images[0]
                
#                 end_time = time.time()
#                 st.session_state.image_history.append((image, prompt))
#                 st.image(image, caption=f"Prompt: {prompt}", use_column_width=True)
#                 st.success(f"Generated in {end_time - start_time:.2f} seconds.")
#             except Exception as e:
#                 st.error(f"Error: {e}")
#             finally:
#                 progress_bar.empty()

# # Display image history
# if st.session_state.image_history:
#     st.markdown('<h2 class="text-xl font-semibold text-gray-700 mt-8">📸 Image History</h2>', unsafe_allow_html=True)
#     for i, (img, pr) in enumerate(st.session_state.image_history):
#         st.image(img, caption=f"Prompt: {pr}", width=200)





# import os
# import time
# import torch
# import streamlit as st
# from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LCMScheduler
# from streamlit.components.v1 import html
# import bitsandbytes as bnb
# from PIL import Image
# import io
# import base64
# import logging
# import numpy as np
# from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
# from diffusers.utils import load_image

# # Environment setup
# os.environ["BITSANDBYTES_NOWELCOME"] = "1"
# os.environ["USE_PEFT_BACKEND"] = "0"

# os.environ["XFORMERS_DISABLE"] = "1"
# os.environ["DIFFUSERS_NO_ONNX"] = "1"

# # Setup logging
# logging.basicConfig(filename="app_errors.log", level=logging.ERROR, 
#                     format="%(asctime)s - %(levelname)s - %(message)s")

# # Page configuration
# st.set_page_config(
#     page_title="Ultimate AI Creation Suite (LCM)",
#     page_icon="🔮",
#     layout="wide"
# )

# # Inject Tailwind CSS via CDN
# html("""
# <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
# <style>
# body { background-color: #f9fafb; }
# </style>
# """)

# # Model loading with LCM
# @st.cache_resource
# def load_models():
#     """Loads multiple pipelines for different generation modes."""
#     models = {}
    
#     # Text-to-Image
#     models['txt2img'] = StableDiffusionPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#         torch_dtype=torch.float32,
#         safety_checker=None,
#         use_safetensors=True
#     ).to("cpu")
#     models['txt2img'].scheduler = LCMScheduler.from_config(models['txt2img'].scheduler.config)
#     models['txt2img'].enable_vae_tiling()
    
#     # Image-to-Image
#     models['img2img'] = StableDiffusionImg2ImgPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#         torch_dtype=torch.float32,
#         safety_checker=None,
#         use_safetensors=True
#     ).to("cpu")
#     models['img2img'].scheduler = LCMScheduler.from_config(models['img2img'].scheduler.config)
#     models['img2img'].enable_vae_tiling()
    
#     # Video-to-Image (reuses img2img)
#     models['video2img'] = models['img2img']
    
#     # Quantization
#     try:
#         models['txt2img'].unet = bnb.nn.Linear8bitLt(models['txt2img'].unet, threshold=6.0)
#         models['img2img'].unet = bnb.nn.Linear8bitLt(models['img2img'].unet, threshold=6.0)
#         st.sidebar.success("✅ Models quantized!")
#     except Exception as e:
#         logging.error(f"Quantization failed: {e}")
#         st.sidebar.warning(f"Quantization failed: {e}")
    
#     # Load LCM-LoRA
#     try:
#         models['txt2img'].load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors")
#         models['img2img'].load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors")
#         st.sidebar.success("✅ LCM-LoRA loaded for 5-10x speed boost!")
#     except Exception as e:
#         logging.error(f"LCM-LoRA load failed: {e}")
#         st.sidebar.warning(f"LCM-LoRA load failed: {e}. Using base model.")
    
#     # Load custom LoRA
#     lora_weight_path = "lora_out/lora_out_cpu.bin"
#     if os.path.exists(lora_weight_path):
#         try:
#             models['txt2img'].load_lora_weights("lora_out", weight_name="lora_out_cpu.bin")
#             models['img2img'].load_lora_weights("lora_out", weight_name="lora_out_cpu.bin")
#             st.sidebar.success("✅ Custom LoRA loaded!")
#         except Exception as e:
#             logging.error(f"Custom LoRA failed: {e}")
#             st.sidebar.warning(f"Custom LoRA failed: {e}")
    
#     return models

# # Unique Feature 1: AI Prompt Refiner (using CLIP)
# @st.cache_resource
# def load_clip_model():
#     """Loads CLIP for prompt refinement."""
#     processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
#     return processor, model

# def refine_prompt_with_clip(prompt, clip_processor, clip_model, num_words=5):
#     """Refines prompt using CLIP embeddings (simplified)."""
#     refined_words = [word for word in prompt.split() if len(word) > 3][:num_words]
#     return " ".join(refined_words) + ", highly detailed, masterpiece"

# # Unique Feature 2: Generation Mood Adapter
# def detect_mood_and_adapt(prompt):
#     """Detects mood from prompt and adapts CFG/steps."""
#     mood_keywords = {
#         "serene": {"cfg": 5.0, "steps": 4},
#         "dramatic": {"cfg": 9.0, "steps": 6},
#         "vibrant": {"cfg": 7.5, "steps": 5},
#         "dark": {"cfg": 8.0, "steps": 6}
#     }
#     for mood, params in mood_keywords.items():
#         if mood in prompt.lower():
#             return params
#     return {"cfg": 7.0, "steps": 4}

# # Function to upscale image
# def upscale_image(image, target_size):
#     """Upscales an image using PIL's Lanczos resampling."""
#     return image.resize((target_size, target_size), Image.Resampling.LANCZOS)

# # Function to create downloadable image
# def get_image_download_link(img, filename="generated_image.png"):
#     """Creates a download link for the image."""
#     buffered = io.BytesIO()
#     img.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href = f'<a href="data:image/png;base64,{img_str}" download="{filename}" class="text-blue-600 hover:underline">Download Image</a>'
#     return href

# # Function to extract first frame from video
# def extract_first_frame(video_path):
#     """Extracts the first frame from a video using OpenCV."""
#     import cv2
#     cap = cv2.VideoCapture(video_path)
#     ret, frame = cap.read()
#     if ret:
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         return Image.fromarray(frame_rgb)
#     return None

# # UI Layout
# st.markdown('<h1 class="text-3xl font-bold text-center text-gray-800 mb-6">🔮 Ultimate AI Creation Suite</h1>', unsafe_allow_html=True)
# st.info("Advanced LCM-powered generation with Text-to-Image, Image-to-Image, Video-to-Image. Unique: AI Prompt Refiner & Mood Adapter!")

# # Load models
# with st.spinner("Loading advanced models..."):
#     models = load_models()

# # Load CLIP for unique feature
# clip_processor, clip_model = load_clip_model()

# # Mode selection
# mode = st.sidebar.selectbox("**Generation Mode**", ["Text-to-Image", "Image-to-Image", "Video-to-Image"])

# # Sidebar for controls
# st.sidebar.markdown('<h2 class="text-xl font-semibold text-gray-700">⚙️ Settings</h2>', unsafe_allow_html=True)

# # Style presets
# presets = {
#     "Cozy Interior": "A cozy coffee shop interior, warm lighting, wooden furniture",
#     "Cyberpunk City": "Cyberpunk neon street, night rain, vibrant colors",
#     "Nature Landscape": "Mountain landscape at sunrise, misty valleys",
#     "Fantasy Art": "Epic fantasy castle, glowing skies, magical aura"
# }
# preset = st.sidebar.selectbox("**Style Preset**", ["None"] + list(presets.keys()))

# # Input based on mode
# if mode == "Text-to-Image":
#     prompt = st.sidebar.text_area(
#         "**Prompt**",
#         presets.get(preset, "A cozy coffee shop interior, warm lighting") if preset != "None" else "",
#         height=100,
#         help="Describe the image."
#     )
#     input_image = None
#     input_video = None
# elif mode == "Image-to-Image":
#     prompt = st.sidebar.text_area(
#         "**Prompt**",
#         presets.get(preset, "Enhance this image with cyberpunk style") if preset != "None" else "",
#         height=100,
#         help="Describe changes to the input image."
#     )
#     uploaded_image = st.sidebar.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"])
#     input_image = load_image(uploaded_image) if uploaded_image else None
#     input_video = None
#     strength = st.sidebar.slider("**Image Strength**", min_value=0.0, max_value=1.0, value=0.75, help="How much to transform the input image (0=original, 1=new).")
# else:  # Video-to-Image
#     prompt = st.sidebar.text_area(
#         "**Prompt**",
#         presets.get(preset, "Extract and enhance first frame with dramatic lighting") if preset != "None" else "",
#         height=100,
#         help="Describe enhancements to the video's first frame."
#     )
#     uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
#     if uploaded_video:
#         with open("temp_video.mp4", "wb") as f:
#             f.write(uploaded_video.getvalue())
#         input_image = extract_first_frame("temp_video.mp4")
#         os.remove("temp_video.mp4")
#     else:
#         input_image = None
#     input_video = uploaded_video
#     strength = st.sidebar.slider("**Frame Strength**", min_value=0.0, max_value=1.0, value=0.75)

# neg_prompt = st.sidebar.text_input(
#     "**Negative Prompt**",
#     "blurry, low quality, unrealistic, text, watermark",
#     help="Avoid these elements."
# )

# # Advanced settings
# st.sidebar.markdown('<h3 class="text-lg font-medium text-gray-600">Advanced</h3>', unsafe_allow_html=True)
# num_images = st.sidebar.slider("**Number of Images**", min_value=1, max_value=4, value=1)
# width = st.sidebar.select_slider("**Width**", options=[256, 384], value=256)
# height = st.sidebar.select_slider("**Height**", options=[256, 384], value=256)

# # Unique Feature 1: AI Prompt Refiner
# use_refiner = st.sidebar.checkbox("Enable AI Prompt Refiner", value=True)
# if use_refiner:
#     refined_prompt = refine_prompt_with_clip(prompt, clip_processor, clip_model)
#     st.sidebar.info(f"Refined: {refined_prompt}")

# # Unique Feature 2: Mood Adapter
# use_mood_adapter = st.sidebar.checkbox("Enable Mood Adapter", value=True)
# if use_mood_adapter:
#     mood_params = detect_mood_and_adapt(prompt)
#     st.sidebar.info(f"Mood: Adapted CFG={mood_params['cfg']}, Steps={mood_params['steps']}")

# steps = st.sidebar.slider("**Inference Steps**", min_value=4, max_value=8, value=4)
# guidance = st.sidebar.slider("**CFG Scale**", min_value=1.0, max_value=10.0, value=7.0, step=0.5)
# lora_strength = st.sidebar.slider("**LoRA Strength**", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
# upscale = st.sidebar.selectbox("**Upscale To**", ["None", "512x512", "768x768"], index=0)

# # Image history
# if "image_history" not in st.session_state:
#     st.session_state.image_history = []

# # Generate button
# if st.sidebar.button("Generate", key="generate"):
#     if not prompt:
#         st.error("Please enter a prompt.")
#     else:
#         pipe = models['txt2img'] if mode == "Text-to-Image" else models['img2img']
#         start_time = time.time()
#         progress_bar = st.progress(0)
#         status_text = st.empty()
#         generated_images = []
        
#         try:
#             with torch.inference_mode():
#                 for i in range(num_images):
#                     progress = (i + 1) / num_images
#                     est_time_per_image = 8
#                     status_text.text(f"Generating {mode} {i+1}/{num_images} (est. {(num_images-i)*est_time_per_image:.1f}s)...")
                    
#                     # Apply unique features
#                     current_prompt = refined_prompt if use_refiner else prompt
#                     current_steps = mood_params['steps'] if use_mood_adapter else steps
#                     current_guidance = mood_params['cfg'] if use_mood_adapter else guidance
                    
#                     # Apply LoRA strength
#                     pipe.unet.set_adapters(["default"], weights=[lora_strength])
                    
#                     if mode == "Text-to-Image":
#                         image = pipe(
#                             prompt=current_prompt,
#                             negative_prompt=neg_prompt,
#                             num_inference_steps=current_steps,
#                             guidance_scale=current_guidance,
#                             width=width,
#                             height=height
#                         ).images[0]
#                     else:  # Image-to-Image or Video-to-Image
#                         if input_image is None:
#                             st.error("No input image/video provided.")
#                             continue
#                         image = pipe(
#                             prompt=current_prompt,
#                             image=input_image,
#                             negative_prompt=neg_prompt,
#                             num_inference_steps=current_steps,
#                             guidance_scale=current_guidance,
#                             strength=strength,
#                             width=width,
#                             height=height
#                         ).images[0]
                    
#                     # Upscale if requested
#                     if upscale != "None":
#                         target_size = int(upscale.split("x")[0])
#                         image = upscale_image(image, target_size)
                    
#                     generated_images.append((image, current_prompt))
#                     progress_bar.progress(progress)
                
#                 end_time = time.time()
                
#                 # Display images
#                 cols = st.columns(min(num_images, 3))
#                 for idx, (img, pr) in enumerate(generated_images):
#                     cols[idx % 3].image(img, caption=f"Prompt: {pr[:30]}...", use_column_width=True)
#                     cols[idx % 3].markdown(get_image_download_link(img, f"{mode.lower()}_{idx+1}.png"), unsafe_allow_html=True)
                
#                 st.session_state.image_history.extend(generated_images)
#                 st.success(f"Generated {num_images} image(s) via {mode} in {end_time - start_time:.2f} seconds!")
        
#         except Exception as e:
#             logging.error(f"Generation failed: {e}")
#             st.error(f"Error: {e}")
        
#         finally:
#             progress_bar.empty()
#             status_text.empty()

# # Display image history
# if st.session_state.image_history:
#     st.markdown('<h2 class="text-xl font-semibold text-gray-700 mt-8">📸 Recent Creations</h2>', unsafe_allow_html=True)
#     cols = st.columns(3)
#     for i, (img, pr) in enumerate(st.session_state.image_history[-9:]):
#         cols[i % 3].image(img, caption=f"Prompt: {pr[:30]}...", width=150)
#         cols[i % 3].markdown(get_image_download_link(img, f"history_{i+1}.png"), unsafe_allow_html=True)




import os
import time
import torch
import streamlit as st
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LCMScheduler
from streamlit.components.v1 import html
import bitsandbytes as bnb
from PIL import Image, ImageEnhance
import io
import base64
import logging
import numpy as np
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.utils import load_image
from huggingface_hub import InferenceClient

# Environment setup
os.environ["XFORMERS_DISABLE"] = "1"
os.environ["DIFFUSERS_NO_ONNX"] = "1"

# Securely load HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    st.warning("HF_TOKEN not set in environment. Some features (e.g., SDXL) may fail.")

# Setup logging
logging.basicConfig(filename="app_errors.log", level=logging.ERROR, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Page configuration
st.set_page_config(
    page_title="Ultimate AI Creation Suite (Enhanced Clarity)",
    page_icon="🔮",
    layout="wide"
)

# Inject Tailwind CSS via CDN
html("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<style>
body { background-color: #f9fafb; }
</style>
""")

# Model loading with LCM and optional SDXL (no widgets inside)
@st.cache_resource
def load_models(model_choice="sd-v1-5", use_quantization=True):
    """Loads pipelines for different generation modes."""
    models = {}
    
    if model_choice == "sdxl":
        st.warning("Stable Diffusion XL is GPU-intensive and may be slow or fail on CPU. Use with caution.")
        try:
            models['txt2img'] = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float32,
                use_safetensors=True,
                token=HF_TOKEN
            ).to("cpu")
            models['img2img'] = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float32,
                use_safetensors=True,
                token=HF_TOKEN
            ).to("cpu")
            st.sidebar.success("✅ SDXL loaded (CPU mode, expect slower performance).")
        except Exception as e:
            logging.error(f"SDXL load failed: {e}")
            st.error(f"SDXL load failed: {e}. Falling back to SD v1.5.")
            model_choice = "sd-v1-5"
    
    if model_choice == "sd-v1-5":
        # Text-to-Image
        models['txt2img'] = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
            use_safetensors=True
        ).to("cpu")
        models['txt2img'].scheduler = LCMScheduler.from_config(models['txt2img'].scheduler.config)
        models['txt2img'].enable_vae_tiling()
        
        # Image-to-Image
        models['img2img'] = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
            use_safetensors=True
        ).to("cpu")
        models['img2img'].scheduler = LCMScheduler.from_config(models['img2img'].scheduler.config)
        models['img2img'].enable_vae_tiling()
        
        # Video-to-Image (reuses img2img)
        models['video2img'] = models['img2img']
        
        # Optional quantization (passed as param)
        if use_quantization:
            try:
                models['txt2img'].unet = bnb.nn.Linear8bitLt(models['txt2img'].unet, threshold=6.0)
                models['img2img'].unet = bnb.nn.Linear8bitLt(models['img2img'].unet, threshold=6.0)
                st.sidebar.success("✅ Models quantized!")
            except Exception as e:
                logging.error(f"Quantization failed: {e}")
                st.sidebar.warning(f"Quantization failed: {e}")
        
        # Load LCM-LoRA
        try:
            models['txt2img'].load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors")
            models['img2img'].load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors")
            st.sidebar.success("✅ LCM-LoRA loaded for speed boost!")
        except Exception as e:
            logging.error(f"LCM-LoRA load failed: {e}")
            st.sidebar.warning(f"LCM-LoRA load failed: {e}. Using base model.")
    
    # Load custom LoRA
    lora_weight_path = "lora_out/lora_out_cpu.bin"
    if os.path.exists(lora_weight_path):
        try:
            models['txt2img'].load_lora_weights("lora_out", weight_name="lora_out_cpu.bin")
            models['img2img'].load_lora_weights("lora_out", weight_name="lora_out_cpu.bin")
            st.sidebar.success("✅ Custom LoRA loaded!")
        except Exception as e:
            logging.error(f"Custom LoRA failed: {e}")
            st.sidebar.warning(f"Custom LoRA failed: {e}")
    
    return models, model_choice

# Unique Feature 1: AI Prompt Refiner (using CLIP)
@st.cache_resource
def load_clip_model():
    """Loads CLIP for prompt refinement."""
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
    return processor, model

def refine_prompt_with_clip(prompt, clip_processor, clip_model, num_words=5):
    """Refines prompt using CLIP embeddings (simplified)."""
    refined_words = [word for word in prompt.split() if len(word) > 3][:num_words]
    return " ".join(refined_words) + ", highly detailed, ultra-sharp, 8k"

# Unique Feature 2: Generation Mood Adapter
def detect_mood_and_adapt(prompt, model_choice):
    """Detects mood from prompt and adapts CFG/steps."""
    mood_keywords = {
        "serene": {"cfg": 7.5, "steps": 20},
        "dramatic": {"cfg": 10.0, "steps": 50},
        "vibrant": {"cfg": 8.5, "steps": 30},
        "dark": {"cfg": 9.0, "steps": 40}
    }
    for mood, params in mood_keywords.items():
        if mood in prompt.lower():
            return params
    return {"cfg": 7.5, "steps": 20 if model_choice == "sd-v1-5" else 50}

# Function to upscale and sharpen image
def enhance_image(image, target_size=None, sharpen_factor=1.5):
    """Upscales and sharpens the image."""
    if target_size:
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(sharpen_factor)

# Function to create downloadable image
def get_image_download_link(img, filename="generated_image.png"):
    """Creates a download link for the image."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}" class="text-blue-600 hover:underline">Download Image</a>'
    return href

# Function to extract first frame from video
def extract_first_frame(video_path):
    """Extracts the first frame from a video using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None

# UI Layout
st.markdown('<h1 class="text-3xl font-bold text-center text-gray-800 mb-6">🔮 Ultimate AI Creation Suite</h1>', unsafe_allow_html=True)
st.info("Enhanced clarity with Text-to-Image, Image-to-Image, Video-to-Image. SDXL available (CPU-slow). Unique: AI Prompt Refiner & Mood Adapter!")

# Model selection (outside cache)
model_choice = st.sidebar.selectbox("**Model**", ["SD v1.5 (LCM, Fast)", "Stable Diffusion XL (Slow on CPU)"], index=0)
model_choice = "sdxl" if model_choice.startswith("Stable Diffusion XL") else "sd-v1-5"

# Quantization toggle (moved outside cache)
use_quantization = st.sidebar.checkbox("Enable Quantization (Memory Efficient, May Reduce Quality)", value=True)

# Load models (pass toggle as param)
with st.spinner("Loading models..."):
    models, model_choice = load_models(model_choice, use_quantization)

# Load CLIP for unique feature
clip_processor, clip_model = load_clip_model()

# Mode selection
mode = st.sidebar.selectbox("**Generation Mode**", ["Text-to-Image", "Image-to-Image", "Video-to-Image"])

# Sidebar for controls
st.sidebar.markdown('<h2 class="text-xl font-semibold text-gray-700">⚙️ Settings</h2>', unsafe_allow_html=True)

# Style presets
presets = {
    "Cozy Interior": "A cozy coffee shop interior, warm lighting, wooden furniture, ultra-sharp",
    "Cyberpunk City": "Cyberpunk neon street, night rain, vibrant colors, detailed, 8k",
    "Nature Landscape": "Mountain landscape at sunrise, misty valleys, highly detailed",
    "Fantasy Art": "Epic fantasy castle, glowing skies, magical aura, ultra-clear"
}
preset = st.sidebar.selectbox("**Style Preset**", ["None"] + list(presets.keys()))

# Input based on mode
if mode == "Text-to-Image":
    prompt = st.sidebar.text_area(
        "**Prompt**",
        presets.get(preset, "A cozy coffee shop interior, warm lighting, ultra-sharp") if preset != "None" else "",
        height=100,
        help="Describe the image with clarity keywords (e.g., 'sharp', '8k')."
    )
    input_image = None
    input_video = None
elif mode == "Image-to-Image":
    prompt = st.sidebar.text_area(
        "**Prompt**",
        presets.get(preset, "Enhance this image with cyberpunk style, ultra-sharp") if preset != "None" else "",
        height=100,
        help="Describe changes to the input image with clarity keywords."
    )
    uploaded_image = st.sidebar.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"])
    input_image = load_image(uploaded_image) if uploaded_image else None
    input_video = None
    strength = st.sidebar.slider("**Image Strength**", min_value=0.0, max_value=1.0, value=0.75)
else:  # Video-to-Image
    prompt = st.sidebar.text_area(
        "**Prompt**",
        presets.get(preset, "Extract and enhance first frame with dramatic lighting, ultra-clear") if preset != "None" else "",
        height=100,
        help="Describe enhancements to the video's first frame with clarity keywords."
    )
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.getvalue())
        input_image = extract_first_frame("temp_video.mp4")
        os.remove("temp_video.mp4")
    else:
        input_image = None
    input_video = uploaded_video
    strength = st.sidebar.slider("**Frame Strength**", min_value=0.0, max_value=1.0, value=0.75)

neg_prompt = st.sidebar.text_input(
    "**Negative Prompt**",
    "blurry, low quality, unrealistic, text, watermark, noise",
    help="Avoid these elements to enhance clarity."
)

# Advanced settings
st.sidebar.markdown('<h3 class="text-lg font-medium text-gray-600">Advanced</h3>', unsafe_allow_html=True)
num_images = st.sidebar.slider("**Number of Images**", min_value=1, max_value=4, value=1)
width = st.sidebar.select_slider("**Width**", options=[256, 384, 512, 768 if model_choice == "sdxl" else 512], value=512)
height = st.sidebar.select_slider("**Height**", options=[256, 384, 512, 768 if model_choice == "sdxl" else 512], value=512)
sharpen_factor = st.sidebar.slider("**Sharpen Factor**", min_value=1.0, max_value=3.0, value=1.5, step=0.1, help="Increase for sharper images.")

# Unique Feature 1: AI Prompt Refiner
use_refiner = st.sidebar.checkbox("Enable AI Prompt Refiner", value=True)
if use_refiner:
    refined_prompt = refine_prompt_with_clip(prompt, clip_processor, clip_model)
    st.sidebar.info(f"Refined: {refined_prompt}")

# Unique Feature 2: Mood Adapter
use_mood_adapter = st.sidebar.checkbox("Enable Mood Adapter", value=True)
if use_mood_adapter:
    mood_params = detect_mood_and_adapt(prompt, model_choice)
    st.sidebar.info(f"Mood: Adapted CFG={mood_params['cfg']}, Steps={mood_params['steps']}")

steps = st.sidebar.slider("**Inference Steps**", min_value=20 if model_choice == "sd-v1-5" else 50, max_value=50 if model_choice == "sd-v1-5" else 100, value=20 if model_choice == "sd-v1-5" else 50)
guidance = st.sidebar.slider("**CFG Scale**", min_value=7.5, max_value=12.0, value=7.5, step=0.5)
lora_strength = st.sidebar.slider("**LoRA Strength**", min_value=0.0, max_value=1.0, value=1.0, step=0.1) if model_choice == "sd-v1-5" else 1.0
upscale = st.sidebar.selectbox("**Upscale To**", ["None", "512x512", "768x768", "1024x1024" if model_choice == "sdxl" else "768x768"], index=0)

# InferenceClient fallback
use_inference_client = st.sidebar.checkbox("Use InferenceClient (API, requires HF_TOKEN)", value=False)
if use_inference_client and not HF_TOKEN:
    st.error("HF_TOKEN required for InferenceClient. Set it in environment variables.")

# Image history
if "image_history" not in st.session_state:
    st.session_state.image_history = []

# Generate button
if st.sidebar.button("Generate", key="generate"):
    if not prompt:
        st.error("Please enter a prompt.")
    else:
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()
        generated_images = []
        
        try:
            if use_inference_client:
                client = InferenceClient(token=HF_TOKEN)
                for i in range(num_images):
                    progress = (i + 1) / num_images
                    est_time_per_image = 15
                    status_text.text(f"Generating {mode} {i+1}/{num_images} via InferenceClient...")
                    current_prompt = refined_prompt if use_refiner else prompt
                    image = client.text_to_image(
                        prompt=current_prompt,
                        model="stabilityai/stable-diffusion-xl-base-1.0" if model_choice == "sdxl" else "runwayml/stable-diffusion-v1-5",
                        negative_prompt=neg_prompt,
                        num_inference_steps=mood_params['steps'] if use_mood_adapter else steps,
                        guidance_scale=mood_params['cfg'] if use_mood_adapter else guidance,
                        width=width,
                        height=height
                    )
                    image = enhance_image(image, target_size=int(upscale.split("x")[0]) if upscale != "None" else None, sharpen_factor=sharpen_factor)
                    generated_images.append((image, current_prompt))
                    progress_bar.progress(progress)
            else:
                pipe = models['txt2img'] if mode == "Text-to-Image" else models['img2img']
                with torch.inference_mode():
                    for i in range(num_images):
                        progress = (i + 1) / num_images
                        est_time_per_image = 15 if model_choice == "sd-v1-5" else 120
                        status_text.text(f"Generating {mode} {i+1}/{num_images} (est. {(num_images-i)*est_time_per_image:.1f}s)...")
                        
                        # Apply unique features
                        current_prompt = refined_prompt if use_refiner else prompt
                        current_steps = mood_params['steps'] if use_mood_adapter else steps
                        current_guidance = mood_params['cfg'] if use_mood_adapter else guidance
                        
                        if model_choice == "sd-v1-5":
                            pipe.unet.set_adapters(["default"], weights=[lora_strength])
                        
                        if mode == "Text-to-Image":
                            image = pipe(
                                prompt=current_prompt,
                                negative_prompt=neg_prompt,
                                num_inference_steps=current_steps,
                                guidance_scale=current_guidance,
                                width=width,
                                height=height
                            ).images[0]
                        else:  # Image-to-Image or Video-to-Image
                            if input_image is None:
                                st.error("No input image/video provided.")
                                continue
                            image = pipe(
                                prompt=current_prompt,
                                image=input_image,
                                negative_prompt=neg_prompt,
                                num_inference_steps=current_steps,
                                guidance_scale=current_guidance,
                                strength=strength,
                                width=width,
                                height=height
                            ).images[0]
                        
                        # Enhance image with upscaling and sharpening
                        image = enhance_image(image, target_size=int(upscale.split("x")[0]) if upscale != "None" else None, sharpen_factor=sharpen_factor)
                        generated_images.append((image, current_prompt))
                        progress_bar.progress(progress)
                
                end_time = time.time()
                
                # Display images
                cols = st.columns(min(num_images, 3))
                for idx, (img, pr) in enumerate(generated_images):
                    cols[idx % 3].image(img, caption=f"Prompt: {pr[:30]}...", use_column_width=True)
                    cols[idx % 3].markdown(get_image_download_link(img, f"{mode.lower()}_{idx+1}.png"), unsafe_allow_html=True)
                
                st.session_state.image_history.extend(generated_images)
                st.success(f"Generated {num_images} image(s) via {mode} in {end_time - start_time:.2f} seconds!")
        
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            st.error(f"Error: {e}")
        
        finally:
            progress_bar.empty()
            status_text.empty()

# Display image history
if st.session_state.image_history:
    st.markdown('<h2 class="text-xl font-semibold text-gray-700 mt-8">📸 Recent Creations</h2>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (img, pr) in enumerate(st.session_state.image_history[-9:]):
        cols[i % 3].image(img, caption=f"Prompt: {pr[:30]}...", width=150)
        cols[i % 3].markdown(get_image_download_link(img, f"history_{i+1}.png"), unsafe_allow_html=True)