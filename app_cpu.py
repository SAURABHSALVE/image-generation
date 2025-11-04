# import os
# import time
# import torch
# import streamlit as st
# from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LCMScheduler
# from streamlit.components.v1 import html
# import bitsandbytes as bnb
# from PIL import Image, ImageEnhance
# import io
# import base64
# import logging
# import numpy as np
# from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
# from diffusers.utils import load_image
# from huggingface_hub import InferenceClient

# # Environment setup
# os.environ["XFORMERS_DISABLE"] = "1"
# os.environ["DIFFUSERS_NO_ONNX"] = "1"

# # Securely load HF_TOKEN
# HF_TOKEN = os.environ.get("HF_TOKEN", "")
# if not HF_TOKEN:
#     st.warning("HF_TOKEN not set in environment. Some features (e.g., SDXL) may fail.")

# # Setup logging
# logging.basicConfig(filename="app_errors.log", level=logging.ERROR, 
#                     format="%(asctime)s - %(levelname)s - %(message)s")

# # Page configuration
# st.set_page_config(
#     page_title="Ultimate AI Creation Suite (Enhanced Clarity)",
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

# # Model loading with LCM and optional SDXL (no widgets inside)
# @st.cache_resource
# def load_models(model_choice="sd-v1-5", use_quantization=True):
#     """Loads pipelines for different generation modes."""
#     models = {}
    
#     if model_choice == "sdxl":
#         st.warning("Stable Diffusion XL is GPU-intensive and may be slow or fail on CPU. Use with caution.")
#         try:
#             models['txt2img'] = StableDiffusionPipeline.from_pretrained(
#                 "stabilityai/stable-diffusion-xl-base-1.0",
#                 torch_dtype=torch.float32,
#                 use_safetensors=True,
#                 token=HF_TOKEN
#             ).to("cpu")
#             models['img2img'] = StableDiffusionImg2ImgPipeline.from_pretrained(
#                 "stabilityai/stable-diffusion-xl-base-1.0",
#                 torch_dtype=torch.float32,
#                 use_safetensors=True,
#                 token=HF_TOKEN
#             ).to("cpu")
#             st.sidebar.success("✅ SDXL loaded (CPU mode, expect slower performance).")
#         except Exception as e:
#             logging.error(f"SDXL load failed: {e}")
#             st.error(f"SDXL load failed: {e}. Falling back to SD v1.5.")
#             model_choice = "sd-v1-5"
    
#     if model_choice == "sd-v1-5":
#         # Text-to-Image
#         models['txt2img'] = StableDiffusionPipeline.from_pretrained(
#             "runwayml/stable-diffusion-v1-5",
#             torch_dtype=torch.float32,
#             safety_checker=None,
#             use_safetensors=True
#         ).to("cpu")
#         models['txt2img'].scheduler = LCMScheduler.from_config(models['txt2img'].scheduler.config)
#         models['txt2img'].enable_vae_tiling()
        
#         # Image-to-Image
#         models['img2img'] = StableDiffusionImg2ImgPipeline.from_pretrained(
#             "runwayml/stable-diffusion-v1-5",
#             torch_dtype=torch.float32,
#             safety_checker=None,
#             use_safetensors=True
#         ).to("cpu")
#         models['img2img'].scheduler = LCMScheduler.from_config(models['img2img'].scheduler.config)
#         models['img2img'].enable_vae_tiling()
        
#         # Video-to-Image (reuses img2img)
#         models['video2img'] = models['img2img']
        
#         # Optional quantization (passed as param)
#         if use_quantization:
#             try:
#                 models['txt2img'].unet = bnb.nn.Linear8bitLt(models['txt2img'].unet, threshold=6.0)
#                 models['img2img'].unet = bnb.nn.Linear8bitLt(models['img2img'].unet, threshold=6.0)
#                 st.sidebar.success("✅ Models quantized!")
#             except Exception as e:
#                 logging.error(f"Quantization failed: {e}")
#                 st.sidebar.warning(f"Quantization failed: {e}")
        
#         # Load LCM-LoRA
#         try:
#             models['txt2img'].load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors")
#             models['img2img'].load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors")
#             st.sidebar.success("✅ LCM-LoRA loaded for speed boost!")
#         except Exception as e:
#             logging.error(f"LCM-LoRA load failed: {e}")
#             st.sidebar.warning(f"LCM-LoRA load failed: {e}. Using base model.")
    
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
    
#     return models, model_choice

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
#     return " ".join(refined_words) + ", highly detailed, ultra-sharp, 8k"

# # Unique Feature 2: Generation Mood Adapter
# def detect_mood_and_adapt(prompt, model_choice):
#     """Detects mood from prompt and adapts CFG/steps."""
#     mood_keywords = {
#         "serene": {"cfg": 7.5, "steps": 20},
#         "dramatic": {"cfg": 10.0, "steps": 50},
#         "vibrant": {"cfg": 8.5, "steps": 30},
#         "dark": {"cfg": 9.0, "steps": 40}
#     }
#     for mood, params in mood_keywords.items():
#         if mood in prompt.lower():
#             return params
#     return {"cfg": 7.5, "steps": 20 if model_choice == "sd-v1-5" else 50}

# # Function to upscale and sharpen image
# def enhance_image(image, target_size=None, sharpen_factor=1.5):
#     """Upscales and sharpens the image."""
#     if target_size:
#         image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
#     enhancer = ImageEnhance.Sharpness(image)
#     return enhancer.enhance(sharpen_factor)

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
# st.info("Enhanced clarity with Text-to-Image, Image-to-Image, Video-to-Image. SDXL available (CPU-slow). Unique: AI Prompt Refiner & Mood Adapter!")

# # Model selection (outside cache)
# model_choice = st.sidebar.selectbox("**Model**", ["SD v1.5 (LCM, Fast)", "Stable Diffusion XL (Slow on CPU)"], index=0)
# model_choice = "sdxl" if model_choice.startswith("Stable Diffusion XL") else "sd-v1-5"

# # Quantization toggle (moved outside cache)
# use_quantization = st.sidebar.checkbox("Enable Quantization (Memory Efficient, May Reduce Quality)", value=True)

# # Load models (pass toggle as param)
# with st.spinner("Loading models..."):
#     models, model_choice = load_models(model_choice, use_quantization)

# # Load CLIP for unique feature
# clip_processor, clip_model = load_clip_model()

# # Mode selection
# mode = st.sidebar.selectbox("**Generation Mode**", ["Text-to-Image", "Image-to-Image", "Video-to-Image"])

# # Sidebar for controls
# st.sidebar.markdown('<h2 class="text-xl font-semibold text-gray-700">⚙️ Settings</h2>', unsafe_allow_html=True)

# # Style presets
# presets = {
#     "Cozy Interior": "A cozy coffee shop interior, warm lighting, wooden furniture, ultra-sharp",
#     "Cyberpunk City": "Cyberpunk neon street, night rain, vibrant colors, detailed, 8k",
#     "Nature Landscape": "Mountain landscape at sunrise, misty valleys, highly detailed",
#     "Fantasy Art": "Epic fantasy castle, glowing skies, magical aura, ultra-clear"
# }
# preset = st.sidebar.selectbox("**Style Preset**", ["None"] + list(presets.keys()))

# # Input based on mode
# if mode == "Text-to-Image":
#     prompt = st.sidebar.text_area(
#         "**Prompt**",
#         presets.get(preset, "A cozy coffee shop interior, warm lighting, ultra-sharp") if preset != "None" else "",
#         height=100,
#         help="Describe the image with clarity keywords (e.g., 'sharp', '8k')."
#     )
#     input_image = None
#     input_video = None
# elif mode == "Image-to-Image":
#     prompt = st.sidebar.text_area(
#         "**Prompt**",
#         presets.get(preset, "Enhance this image with cyberpunk style, ultra-sharp") if preset != "None" else "",
#         height=100,
#         help="Describe changes to the input image with clarity keywords."
#     )
#     uploaded_image = st.sidebar.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"])
#     input_image = load_image(uploaded_image) if uploaded_image else None
#     input_video = None
#     strength = st.sidebar.slider("**Image Strength**", min_value=0.0, max_value=1.0, value=0.75)
# else:  # Video-to-Image
#     prompt = st.sidebar.text_area(
#         "**Prompt**",
#         presets.get(preset, "Extract and enhance first frame with dramatic lighting, ultra-clear") if preset != "None" else "",
#         height=100,
#         help="Describe enhancements to the video's first frame with clarity keywords."
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
#     "blurry, low quality, unrealistic, text, watermark, noise",
#     help="Avoid these elements to enhance clarity."
# )

# # Advanced settings
# st.sidebar.markdown('<h3 class="text-lg font-medium text-gray-600">Advanced</h3>', unsafe_allow_html=True)
# num_images = st.sidebar.slider("**Number of Images**", min_value=1, max_value=4, value=1)
# width = st.sidebar.select_slider("**Width**", options=[256, 384, 512, 768 if model_choice == "sdxl" else 512], value=512)
# height = st.sidebar.select_slider("**Height**", options=[256, 384, 512, 768 if model_choice == "sdxl" else 512], value=512)
# sharpen_factor = st.sidebar.slider("**Sharpen Factor**", min_value=1.0, max_value=3.0, value=1.5, step=0.1, help="Increase for sharper images.")

# # Unique Feature 1: AI Prompt Refiner
# use_refiner = st.sidebar.checkbox("Enable AI Prompt Refiner", value=True)
# if use_refiner:
#     refined_prompt = refine_prompt_with_clip(prompt, clip_processor, clip_model)
#     st.sidebar.info(f"Refined: {refined_prompt}")

# # Unique Feature 2: Mood Adapter
# use_mood_adapter = st.sidebar.checkbox("Enable Mood Adapter", value=True)
# if use_mood_adapter:
#     mood_params = detect_mood_and_adapt(prompt, model_choice)
#     st.sidebar.info(f"Mood: Adapted CFG={mood_params['cfg']}, Steps={mood_params['steps']}")

# steps = st.sidebar.slider("**Inference Steps**", min_value=20 if model_choice == "sd-v1-5" else 50, max_value=50 if model_choice == "sd-v1-5" else 100, value=20 if model_choice == "sd-v1-5" else 50)
# guidance = st.sidebar.slider("**CFG Scale**", min_value=7.5, max_value=12.0, value=7.5, step=0.5)
# lora_strength = st.sidebar.slider("**LoRA Strength**", min_value=0.0, max_value=1.0, value=1.0, step=0.1) if model_choice == "sd-v1-5" else 1.0
# upscale = st.sidebar.selectbox("**Upscale To**", ["None", "512x512", "768x768", "1024x1024" if model_choice == "sdxl" else "768x768"], index=0)

# # InferenceClient fallback
# use_inference_client = st.sidebar.checkbox("Use InferenceClient (API, requires HF_TOKEN)", value=False)
# if use_inference_client and not HF_TOKEN:
#     st.error("HF_TOKEN required for InferenceClient. Set it in environment variables.")

# # Image history
# if "image_history" not in st.session_state:
#     st.session_state.image_history = []

# # Generate button
# if st.sidebar.button("Generate", key="generate"):
#     if not prompt:
#         st.error("Please enter a prompt.")
#     else:
#         start_time = time.time()
#         progress_bar = st.progress(0)
#         status_text = st.empty()
#         generated_images = []
        
#         try:
#             if use_inference_client:
#                 client = InferenceClient(token=HF_TOKEN)
#                 for i in range(num_images):
#                     progress = (i + 1) / num_images
#                     est_time_per_image = 15
#                     status_text.text(f"Generating {mode} {i+1}/{num_images} via InferenceClient...")
#                     current_prompt = refined_prompt if use_refiner else prompt
#                     image = client.text_to_image(
#                         prompt=current_prompt,
#                         model="stabilityai/stable-diffusion-xl-base-1.0" if model_choice == "sdxl" else "runwayml/stable-diffusion-v1-5",
#                         negative_prompt=neg_prompt,
#                         num_inference_steps=mood_params['steps'] if use_mood_adapter else steps,
#                         guidance_scale=mood_params['cfg'] if use_mood_adapter else guidance,
#                         width=width,
#                         height=height
#                     )
#                     image = enhance_image(image, target_size=int(upscale.split("x")[0]) if upscale != "None" else None, sharpen_factor=sharpen_factor)
#                     generated_images.append((image, current_prompt))
#                     progress_bar.progress(progress)
#             else:
#                 pipe = models['txt2img'] if mode == "Text-to-Image" else models['img2img']
#                 with torch.inference_mode():
#                     for i in range(num_images):
#                         progress = (i + 1) / num_images
#                         est_time_per_image = 15 if model_choice == "sd-v1-5" else 120
#                         status_text.text(f"Generating {mode} {i+1}/{num_images} (est. {(num_images-i)*est_time_per_image:.1f}s)...")
                        
#                         # Apply unique features
#                         current_prompt = refined_prompt if use_refiner else prompt
#                         current_steps = mood_params['steps'] if use_mood_adapter else steps
#                         current_guidance = mood_params['cfg'] if use_mood_adapter else guidance
                        
#                         if model_choice == "sd-v1-5":
#                             pipe.unet.set_adapters(["default"], weights=[lora_strength])
                        
#                         if mode == "Text-to-Image":
#                             image = pipe(
#                                 prompt=current_prompt,
#                                 negative_prompt=neg_prompt,
#                                 num_inference_steps=current_steps,
#                                 guidance_scale=current_guidance,
#                                 width=width,
#                                 height=height
#                             ).images[0]
#                         else:  # Image-to-Image or Video-to-Image
#                             if input_image is None:
#                                 st.error("No input image/video provided.")
#                                 continue
#                             image = pipe(
#                                 prompt=current_prompt,
#                                 image=input_image,
#                                 negative_prompt=neg_prompt,
#                                 num_inference_steps=current_steps,
#                                 guidance_scale=current_guidance,
#                                 strength=strength,
#                                 width=width,
#                                 height=height
#                             ).images[0]
                        
#                         # Enhance image with upscaling and sharpening
#                         image = enhance_image(image, target_size=int(upscale.split("x")[0]) if upscale != "None" else None, sharpen_factor=sharpen_factor)
#                         generated_images.append((image, current_prompt))
#                         progress_bar.progress(progress)
                
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






# import os
# import time
# import torch
# import streamlit as st
# from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LCMScheduler
# from streamlit.components.v1 import html
# import bitsandbytes as bnb
# from PIL import Image, ImageEnhance
# import io
# import base64
# import logging
# from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
# from diffusers.utils import load_image
# from huggingface_hub import InferenceClient

# # Environment setup
# os.environ["XFORMERS_DISABLE"] = "1"
# os.environ["DIFFUSERS_NO_ONNX"] = "1"

# # Securely load HF_TOKEN
# HF_TOKEN = os.environ.get("HF_TOKEN", "")
# if not HF_TOKEN:
#     st.warning("HF_TOKEN not set in environment. Some features (e.g., SDXL) may fail.")

# # Setup logging
# logging.basicConfig(filename="app_errors.log", level=logging.ERROR,
#                     format="%(asctime)s - %(levelname)s - %(message)s")

# # Page configuration
# st.set_page_config(
#     page_title="Ultimate AI Creation Suite",
#     page_icon="🔮",
#     layout="wide"
# )

# # --- ENHANCED UI DESIGN ---
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

# html, body, [class*="st-"] {
#     font-family: 'Inter', sans-serif;
# }

# /* Main background with subtle gradient */
# .stApp {
#     background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
#     color: #212529;
# }

# /* --- Advanced Control Panel Styling --- */
# .control-panel {
#     background: #ffffff;
#     border-radius: 16px;
#     padding: 1.5rem 2rem;
#     margin-bottom: 2rem;
#     border: 1px solid #e9ecef;
#     box-shadow: 0 8px 32px rgba(0, 0, 0, 0.07);
# }

# /* --- Modern Tab Styling --- */
# div[data-baseweb="tab-list"] {
#     background-color: #f8f9fa;
#     padding: 6px;
#     border-radius: 12px;
#     margin-bottom: 1.5rem;
#     box-shadow: inset 0 2px 4px rgba(0,0,0,0.04);
# }

# button[data-baseweb="tab"] {
#     background-color: transparent;
#     border: none;
#     border-radius: 8px;
#     color: #4b5563;
#     font-weight: 600;
#     transition: background-color 0.3s ease, color 0.3s ease;
# }

# button[data-baseweb="tab"][aria-selected="true"] {
#     background-color: #ffffff;
#     color: #4f46e5;
#     box-shadow: 0 2px 8px rgba(0,0,0,0.1);
# }

# button[data-baseweb="tab"]:not([aria-selected="true"]):hover {
#     background-color: #e9ecef;
#     color: #212529;
# }

# button[data-baseweb="tab"] > div {
#     border-bottom: none !important;
# }

# /* Title styling with enhanced gradient and shadow */
# .title-gradient {
#     background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     font-weight: 700;
#     text-shadow: 0 2px 4px rgba(99, 102, 241, 0.3);
#     padding-bottom: 0.5rem;
#     letter-spacing: -0.025em;
# }

# /* Enhanced button styling with glow effect */
# .stButton > button {
#     background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
#     color: white;
#     border: none;
#     border-radius: 12px;
#     padding: 12px 32px;
#     font-weight: 600;
#     font-size: 1rem;
#     transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
#     box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
# }
# .stButton > button:hover {
#     transform: translateY(-2px);
#     filter: brightness(1.05) saturate(1.1);
#     box-shadow: 0 8px 25px rgba(79, 70, 229, 0.5);
# }
# .stButton > button:active {
#     transform: translateY(0);
# }

# /* Enhanced Generated Image Card */
# .image-card {
#     background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
#     border-radius: 16px;
#     padding: 1.5rem;
#     margin-bottom: 1.5rem;
#     border: 1px solid #e9ecef;
#     box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
#     transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
#     position: relative;
#     overflow: hidden;
# }
# .image-card:hover {
#     transform: translateY(-8px) scale(1.02);
#     box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
# }
# .image-card img { border-radius: 12px; width: 100%; }
# .image-card .caption { font-size: 0.95rem; color: #6b7280; margin-top: 1rem; word-wrap: break-word; line-height: 1.5; font-weight: 500; }
# .image-card .download-link a { color: #4f46e5; text-decoration: none; font-weight: 600; padding: 0.5rem 1rem; border-radius: 8px; transition: background 0.2s ease; display: inline-block; margin-top: 0.5rem; }
# .image-card .download-link a:hover { background: rgba(79, 70, 229, 0.1); text-decoration: none; }

# /* Enhanced Placeholder */
# .placeholder {
#     display: flex; flex-direction: column; justify-content: center; align-items: center;
#     min-height: 400px;
#     border: 2px dashed #d1d5db;
#     border-radius: 16px;
#     background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
#     text-align: center;
#     padding: 3rem;
#     transition: all 0.3s ease;
# }
# .placeholder:hover { border-color: #9ca3af; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); }
# .placeholder .icon { font-size: 5rem; animation: pulse 2s infinite; margin-bottom: 1rem; }
# @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
# .placeholder .text { font-size: 1.3rem; color: #6b7280; margin-top: 0; font-weight: 500; line-height: 1.6; }

# /* Section headers */
# h1, h2 { font-weight: 700; color: #1e293b; letter-spacing: -0.025em; }
# h2 { border-bottom: 2px solid #e9ecef; padding-bottom: 0.75rem; margin-bottom: 1.5rem; }
# </style>
# """, unsafe_allow_html=True)

# # Model loading with LCM and optional SDXL (no widgets inside)
# @st.cache_resource
# def load_models(model_choice="sd-v1-5", use_quantization=True):
#     """
#     Loads pipelines for different generation modes.
#     Returns models, the final model choice, and a list of status messages.
#     """
#     models = {}
#     status_messages = []  # Store status messages to be displayed outside the cached function

#     common_args = {
#         "torch_dtype": torch.float32,
#         "use_safetensors": True
#     }
    
#     if model_choice == "sdxl":
#         status_messages.append(("warning", "Stable Diffusion XL is GPU-intensive and may be slow or fail on CPU."))
#         try:
#             models['txt2img'] = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", token=HF_TOKEN, **common_args).to("cpu")
#             models['img2img'] = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", token=HF_TOKEN, **common_args).to("cpu")
#             status_messages.append(("toast", "✅ SDXL loaded (CPU mode)."))
#         except Exception as e:
#             logging.error(f"SDXL load failed: {e}")
#             status_messages.append(("error", f"SDXL load failed: {e}. Falling back to SD v1.5."))
#             model_choice = "sd-v1-5"
    
#     if model_choice == "sd-v1-5":
#         try:
#             pipe_args = {**common_args, "safety_checker": None}
#             if use_quantization:
#                 pipe_args["load_in_8bit"] = True
#                 status_messages.append(("toast", "⚙️ Attempting to load models with 8-bit quantization..."))

#             models['txt2img'] = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", **pipe_args).to("cpu")
#             models['img2img'] = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", **pipe_args).to("cpu")
            
#             if use_quantization:
#                 status_messages.append(("toast", "✅ Models loaded with quantization!"))

#             models['txt2img'].scheduler = LCMScheduler.from_config(models['txt2img'].scheduler.config)
#             models['img2img'].scheduler = LCMScheduler.from_config(models['img2img'].scheduler.config)
            
#             models['txt2img'].enable_vae_tiling()
#             models['img2img'].enable_vae_tiling()

#             models['video2img'] = models['img2img']

#         except Exception as e:
#             logging.error(f"Model loading failed: {e}")
#             status_messages.append(("error", f"Failed to load SD v1.5 models: {e}"))
#             return None, model_choice, status_messages

#         try:
#             models['txt2img'].load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors")
#             models['img2img'].load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors")
#             status_messages.append(("toast", "✅ LCM-LoRA loaded!"))
#         except Exception as e:
#             logging.error(f"LCM-LoRA load failed: {e}")
#             status_messages.append(("warning", f"LCM-LoRA load failed: {e}."))
    
#     lora_weight_path = "lora_out/lora_out_cpu.bin"
#     if os.path.exists(lora_weight_path):
#         try:
#             models['txt2img'].load_lora_weights("lora_out", weight_name="lora_out_cpu.bin")
#             models['img2img'].load_lora_weights("lora_out", weight_name="lora_out_cpu.bin")
#             status_messages.append(("toast", "✅ Custom LoRA loaded!"))
#         except Exception as e:
#             logging.error(f"Custom LoRA failed: {e}")
#             status_messages.append(("warning", f"Custom LoRA failed: {e}"))
    
#     return models, model_choice, status_messages

# @st.cache_resource
# def load_clip_model():
#     """Loads CLIP for prompt refinement."""
#     processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
#     return processor, model

# # --- Helper Functions ---
# def refine_prompt_with_clip(prompt, clip_processor, clip_model, num_words=5):
#     """Refines prompt using CLIP embeddings (simplified)."""
#     prompt = ''.join(c for c in prompt if c.isprintable() or c in [' ', '\n', '\t'])
#     refined_words = [word for word in prompt.split() if len(word) > 3][:num_words]
#     return " ".join(refined_words) + ", highly detailed, ultra-sharp, 8k"

# def detect_mood_and_adapt(prompt, model_choice):
#     """Detects mood from prompt and adapts CFG/steps."""
#     mood_keywords = {"serene": {"cfg": 7.5, "steps": 20}, "dramatic": {"cfg": 10.0, "steps": 50}, "vibrant": {"cfg": 8.5, "steps": 30}, "dark": {"cfg": 9.0, "steps": 40}}
#     for mood, params in mood_keywords.items():
#         if mood in prompt.lower(): return params
#     return {"cfg": 7.5, "steps": 20 if model_choice == "sd-v1-5" else 50}

# def enhance_image(image, target_size=None, sharpen_factor=1.5):
#     """Upscales and sharpens the image."""
#     if target_size: image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
#     return ImageEnhance.Sharpness(image).enhance(sharpen_factor)

# def get_image_download_link(img, filename="generated_image.png"):
#     """Creates a download link for the image."""
#     buffered = io.BytesIO()
#     img.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     return f'<a href="data:image/png;base64,{img_str}" download="{filename}">Download Image</a>'

# def extract_first_frame(video_bytes):
#     """Extracts the first frame from video bytes."""
#     import cv2
#     import numpy as np
#     nparr = np.frombuffer(video_bytes, np.uint8)
#     cap = cv2.VideoCapture()
#     cap.open(nparr)
#     ret, frame = cap.read()
#     if ret:
#         return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     return None


# # --- UI Layout ---
# st.markdown('<h1 class="title-gradient text-3xl md:text-5xl text-center">Ultimate AI Creation Suite</h1>', unsafe_allow_html=True)
# st.info("Unleash your creativity with Text-to-Image, Image-to-Image, and Video-to-Image generation, enhanced with unique AI-driven features.")

# # Load models first
# with st.spinner("Summoning AI models..."):
#     # Default settings
#     initial_model_choice = "sd-v1-5"
#     initial_use_quantization = True
    
#     models, model_choice, status_messages = load_models(initial_model_choice, initial_use_quantization)

# # Display status messages after loading is complete
# for msg_type, msg in status_messages:
#     if msg_type == "toast":
#         st.toast(msg)
#     elif msg_type == "warning":
#         st.warning(msg)
#     elif msg_type == "error":
#         st.error(msg)
        
# if models:
#     clip_processor, clip_model = load_clip_model()
# else:
#     st.error("Model loading failed. The application cannot continue.")
#     st.stop()


# # Main layout with two columns
# col1, col2 = st.columns([1, 2])

# with col1:
#     st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    
#     tab_core, tab_enhance, tab_advanced = st.tabs(["🖌️ Core Settings", "✨ AI Enhancements", "⚙️ Advanced"])

#     # Initialize variables to handle different modes
#     input_image = None
#     strength = 0.75

#     with tab_core:
#         model_choice_display = st.selectbox("**Model**", ["SD v1.5 (LCM, Fast)", "Stable Diffusion XL (Slow on CPU)"], index=0, key="model_select")
#         model_choice = "sdxl" if model_choice_display.startswith("Stable Diffusion XL") else "sd-v1-5"
        
#         mode = st.selectbox("**Generation Mode**", ["Text-to-Image", "Image-to-Image", "Video-to-Image"], key="mode_select")
        
#         prompt = st.text_area("**Prompt**", value="A cozy coffee shop interior", height=100, key="prompt_input")
#         neg_prompt = st.text_input("**Negative Prompt**", "blurry, low quality, unrealistic, text", key="neg_prompt_input")
        
#         if mode == "Image-to-Image":
#             uploaded_image = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"])
#             if uploaded_image:
#                 input_image = load_image(uploaded_image).resize((512, 512))
        
#         elif mode == "Video-to-Image":
#             uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
#             if uploaded_video:
#                 input_image = extract_first_frame(uploaded_video.getvalue())

#     with tab_enhance:
#         presets = {"None": "", "Cozy Interior": "A cozy coffee shop interior...", "Cyberpunk City": "Cyberpunk neon street...", "Nature Landscape": "Mountain landscape at sunrise...", "Fantasy Art": "Epic fantasy castle..."}
#         preset = st.selectbox("**Style Preset**", list(presets.keys()))
#         if preset != "None" and st.session_state.get("prompt_input") != presets[preset]:
#             st.session_state.prompt_input = presets[preset]
#             st.rerun()

#         use_refiner = st.checkbox("Enable AI Prompt Refiner", value=True)
#         if use_refiner and prompt:
#             refined_prompt = refine_prompt_with_clip(prompt, clip_processor, clip_model)
#             st.info(f"Refined: {refined_prompt}")
            
#         use_mood_adapter = st.checkbox("Enable Mood Adapter", value=True)
#         if use_mood_adapter and prompt:
#             mood_params = detect_mood_and_adapt(prompt, model_choice)
#             st.info(f"Mood: CFG={mood_params['cfg']}, Steps={mood_params['steps']}")
    
#     with tab_advanced:
#         if mode != "Text-to-Image":
#             strength = st.slider("**Image/Frame Strength**", 0.0, 1.0, 0.75)
        
#         num_images = st.slider("**Number of Images**", 1, 4, 1)
#         width = st.select_slider("**Width**", [256, 384, 512, 768], value=512)
#         height = st.select_slider("**Height**", [256, 384, 512, 768], value=512)
#         sharpen_factor = st.slider("**Sharpen Factor**", 1.0, 3.0, 1.5, 0.1)
#         upscale = st.selectbox("**Upscale To**", ["None", "512x512", "768x768", "1024x1024"], index=0)
#         steps = st.slider("**Inference Steps**", 4, 50, 20 if model_choice == "sd-v1-5" else 50)
#         guidance = st.slider("**CFG Scale**", 1.0, 12.0, 7.5, 0.5)
        
#         if model_choice == "sd-v1-5":
#             lora_strength = st.slider("**LoRA Strength**", 0.0, 1.0, 1.0, 0.1)
            
#         use_inference_client = st.checkbox("Use InferenceClient (API Fallback)")

#     if st.button("Generate", key="generate", use_container_width=True):
#         if not prompt:
#             st.error("Please enter a prompt.")
#         else:
#             start_time = time.time()
#             progress_bar = st.progress(0, "Starting Generation...")
#             st.session_state.last_generation = []
#             generated_images = []
#             try:
#                 current_prompt = refined_prompt if 'refined_prompt' in locals() and use_refiner else prompt
#                 current_steps = mood_params['steps'] if 'mood_params' in locals() and use_mood_adapter else steps
#                 current_guidance = mood_params['cfg'] if 'mood_params' in locals() and use_mood_adapter else guidance


#                 if use_inference_client:
#                     client = InferenceClient(token=HF_TOKEN)
#                     for i in range(num_images):
#                         progress_bar.progress((i + 1) / num_images, f"Generating via API {i+1}/{num_images}...")
#                         image = client.text_to_image(prompt=current_prompt, model="stabilityai/stable-diffusion-xl-base-1.0" if model_choice == "sdxl" else "runwayml/stable-diffusion-v1-5", negative_prompt=neg_prompt, num_inference_steps=current_steps, guidance_scale=current_guidance, width=width, height=height)
#                         image = enhance_image(image, target_size=int(upscale.split("x")[0]) if upscale != "None" else None, sharpen_factor=sharpen_factor)
#                         generated_images.append((image, current_prompt))
#                 else:
#                     pipe = models['txt2img'] if mode == "Text-to-Image" else models['img2img']
#                     with torch.inference_mode():
#                         for i in range(num_images):
#                             progress_bar.progress((i + 1) / num_images, f"Generating locally {i+1}/{num_images}...")
#                             if model_choice == "sd-v1-5": pipe.unet.set_adapters(["default"], weights=[lora_strength])
                            
#                             if mode == "Text-to-Image":
#                                 image = pipe(prompt=current_prompt, negative_prompt=neg_prompt, num_inference_steps=current_steps, guidance_scale=current_guidance, width=width, height=height).images[0]
#                             else:
#                                 if input_image is None:
#                                     st.error(f"Please upload an {'image' if mode == 'Image-to-Image' else 'video'}.")
#                                     break
#                                 image = pipe(prompt=current_prompt, image=input_image, negative_prompt=neg_prompt, num_inference_steps=current_steps, guidance_scale=current_guidance, strength=strength).images[0]
                            
#                             image = enhance_image(image, target_size=int(upscale.split("x")[0]) if upscale != "None" else None, sharpen_factor=sharpen_factor)
#                             generated_images.append((image, current_prompt))
                
#                 st.success(f"Generated {len(generated_images)} image(s) in {time.time() - start_time:.2f} seconds!")
#                 st.session_state.last_generation = generated_images
#                 if "image_history" not in st.session_state: st.session_state.image_history = []
#                 st.session_state.image_history.extend(generated_images)
#             except Exception as e:
#                 logging.error(f"Generation failed: {e}")
#                 st.error(f"An error occurred during generation: {e}")
#             finally:
#                 progress_bar.empty()
    
#     st.markdown('</div>', unsafe_allow_html=True)


# # Initialize session state
# if "image_history" not in st.session_state: st.session_state.image_history = []
# if "last_generation" not in st.session_state: st.session_state.last_generation = []

# with col2:
#     st.markdown('<h2>✨ Latest Creation</h2>', unsafe_allow_html=True)
#     if st.session_state.last_generation:
#         cols = st.columns(min(len(st.session_state.last_generation), 3))
#         for idx, (img, pr) in enumerate(st.session_state.last_generation):
#             buffered = io.BytesIO()
#             img.save(buffered, format="PNG")
#             img_str = base64.b64encode(buffered.getvalue()).decode()
#             download_html = get_image_download_link(img, f"{mode.lower().replace('-', '_')}_{idx}.png")
            
#             card_html = f"""
#             <div class="image-card">
#                 <img src="data:image/png;base64,{img_str}" alt="Generated Image">
#                 <p class="caption"><b>Prompt:</b> {pr}</p>
#                 <p class="download-link">{download_html}</p>
#             </div>
#             """
#             cols[idx % 3].markdown(card_html, unsafe_allow_html=True)
#     else:
#         st.markdown('<div class="placeholder"><div class="icon">🎨</div><div class="text">Your creations will appear here. <br/> Configure settings and click \'Generate\'!</div></div>', unsafe_allow_html=True)
    
#     if st.session_state.image_history:
#         st.markdown('<h2 class="text-2xl font-semibold mt-8 border-t border-gray-200 pt-6">📸 Recent Creations</h2>', unsafe_allow_html=True)
#         cols = st.columns(4)
#         for i, (img, pr) in enumerate(reversed(st.session_state.image_history[-8:])):
#             buffered = io.BytesIO()
#             img.save(buffered, format="PNG")
#             img_str = base64.b64encode(buffered.getvalue()).decode()
#             download_html = get_image_download_link(img, f"history_{len(st.session_state.image_history)-i}.png")
#             cols[i % 4].markdown(f'<div class="image-card"><img src="data:image/png;base64,{img_str}"><p class="download-link mt-2">{download_html}</p></div>', unsafe_allow_html=True)




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
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.utils import load_image
from huggingface_hub import InferenceClient

# Environment setup
os.environ["XFORMERS_DISABLE"] = "1"
os.environ["DIFFUSERS_NO_ONNX"] = "1"

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

# Model loading with LCM and optional SDXL (no widgets inside)
@st.cache_resource
def load_models(model_choice="sd-v1-5", use_quantization=True):
    """
    Loads pipelines for different generation modes.
    Returns models, the final model choice, and a list of status messages.
    """
    models = {}
    status_messages = []  # Store status messages to be displayed outside the cached function

    common_args = {
        "torch_dtype": torch.float32,
        "use_safetensors": True
    }
    
    if model_choice == "sdxl":
        status_messages.append(("warning", "Stable Diffusion XL is GPU-intensive and may be slow or fail on CPU."))
        try:
            models['txt2img'] = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", token=HF_TOKEN, **common_args).to("cpu")
            models['img2img'] = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", token=HF_TOKEN, **common_args).to("cpu")
            status_messages.append(("toast", "✅ SDXL loaded (CPU mode)."))
        except Exception as e:
            logging.error(f"SDXL load failed: {e}")
            status_messages.append(("error", f"SDXL load failed: {e}. Falling back to SD v1.5."))
            model_choice = "sd-v1-5"
    
    if model_choice == "sd-v1-5":
        try:
            pipe_args = {**common_args, "safety_checker": None}
            if use_quantization:
                pipe_args["load_in_8bit"] = True
                status_messages.append(("toast", "⚙️ Attempting to load models with 8-bit quantization..."))

            models['txt2img'] = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", **pipe_args).to("cpu")
            models['img2img'] = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", **pipe_args).to("cpu")
            
            if use_quantization:
                status_messages.append(("toast", "✅ Models loaded with quantization!"))

            models['txt2img'].scheduler = LCMScheduler.from_config(models['txt2img'].scheduler.config)
            models['img2img'].scheduler = LCMScheduler.from_config(models['img2img'].scheduler.config)
            
            models['txt2img'].enable_vae_tiling()
            models['img2img'].enable_vae_tiling()

            models['video2img'] = models['img2img']

        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            status_messages.append(("error", f"Failed to load SD v1.5 models: {e}"))
            return None, model_choice, status_messages

        try:
            models['txt2img'].load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors")
            models['img2img'].load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors")
            status_messages.append(("toast", "✅ LCM-LoRA loaded!"))
        except Exception as e:
            logging.error(f"LCM-LoRA load failed: {e}")
            status_messages.append(("warning", f"LCM-LoRA load failed: {e}."))
    
    lora_weight_path = "lora_out/lora_out_cpu.bin"
    if os.path.exists(lora_weight_path):
        try:
            models['txt2img'].load_lora_weights("lora_out", weight_name="lora_out_cpu.bin")
            models['img2img'].load_lora_weights("lora_out", weight_name="lora_out_cpu.bin")
            status_messages.append(("toast", "✅ Custom LoRA loaded!"))
        except Exception as e:
            logging.error(f"Custom LoRA failed: {e}")
            status_messages.append(("warning", f"Custom LoRA failed: {e}"))
    
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

def enhance_image(image, target_size=None, sharpen_factor=1.5):
    """Upscales and sharpens the image."""
    if target_size: image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return ImageEnhance.Sharpness(image).enhance(sharpen_factor)

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
    '<h1 class="title-gradient text-3xl md:text-5xl text-center">Ultimate AI Creation Suite</h1>',
    unsafe_allow_html=True
)
st.info(
    "Unleash your creativity with Text-to-Image, Image-to-Image, and Video-to-Image generation, enhanced with unique AI-driven features."
)

# Load models first
with st.spinner("Summoning AI models..."):
    initial_model_choice = "sd-v1-5"
    initial_use_quantization = True
    models, model_choice, status_messages = load_models(initial_model_choice, initial_use_quantization)

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
            "blurry, low quality, unrealistic, text",
            key="neg_prompt_input"
        )

        # --- File upload handling ---
        if mode == "Image-to-Image":
            uploaded_image = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"])
            if uploaded_image:
                input_image = Image.open(uploaded_image).resize((512, 512))

        elif mode == "Video-to-Image":
            uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
            if uploaded_video:
                input_image = extract_first_frame(uploaded_video.getvalue())
                input_image = input_image.resize((512, 512))


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
        if mode != "Text-to-Image":
            strength = st.slider("**Image/Frame Strength**", 0.0, 1.0, 0.75)
        
        num_images = st.slider("**Number of Images**", 1, 4, 1)
        width = st.select_slider("**Width**", [256, 384, 512, 768], value=512)
        height = st.select_slider("**Height**", [256, 384, 512, 768], value=512)
        sharpen_factor = st.slider("**Sharpen Factor**", 1.0, 3.0, 1.5, 0.1)
        upscale = st.selectbox("**Upscale To**", ["None", "512x512", "768x768", "1024x1024"], index=0)
        steps = st.slider("**Inference Steps**", 4, 50, 20 if model_choice == "sd-v1-5" else 50)
        guidance = st.slider("**CFG Scale**", 1.0, 12.0, 7.5, 0.5)
        
        if model_choice == "sd-v1-5":
            lora_strength = st.slider("**LoRA Strength**", 0.0, 1.0, 1.0, 0.1)
            
        use_inference_client = st.checkbox("Use InferenceClient (API Fallback)")

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
                        image = enhance_image(image, target_size=int(upscale.split("x")[0]) if upscale != "None" else None, sharpen_factor=sharpen_factor)
                        generated_images.append((image, current_prompt))
                else:
                    pipe = models['txt2img'] if mode == "Text-to-Image" else models['img2img']
                    with torch.inference_mode():
                        for i in range(num_images):
                            progress_bar.progress((i + 1) / num_images, f"Generating locally {i+1}/{num_images}...")
                            if model_choice == "sd-v1-5": pipe.unet.set_adapters(["default"], weights=[lora_strength])
                            
                            if mode == "Text-to-Image":
                                image = pipe(prompt=current_prompt, negative_prompt=neg_prompt, num_inference_steps=current_steps, guidance_scale=current_guidance, width=width, height=height).images[0]
                            else:
                                if input_image is None:
                                    st.error(f"Please upload an {'image' if mode == 'Image-to-Image' else 'video'}.")
                                    break
                                image = pipe(prompt=current_prompt, image=input_image, negative_prompt=neg_prompt, num_inference_steps=current_steps, guidance_scale=current_guidance, strength=strength).images[0]
                            
                            image = enhance_image(image, target_size=int(upscale.split("x")[0]) if upscale != "None" else None, sharpen_factor=sharpen_factor)
                            generated_images.append((image, current_prompt))
                
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


