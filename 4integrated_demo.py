import os
import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
from PIL import Image
import json
from huggingface_hub import login
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffuser.diffusion import DifussionHandler
from extractor.scene_extractor import extract_scenes_with_gemini

# Đăng nhập Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN", "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")
login(token=HF_TOKEN, add_to_git_credential=True)

# Thiết lập thư mục cache Hugging Face
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# Khởi tạo biến mô hình (lazy loading)
diffuser = None
text_encoder = None
text_encoder_2 = None
vae = None
image_encoder = None
transformer = None
tokenizer = None
tokenizer_2 = None
feature_extractor = None
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# Cấu hình chung
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument('--server', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, required=False)
parser.add_argument('--inbrowser', action='store_true')
args = parser.parse_args()

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60  # Giữ nguyên ngưỡng từ demo_gradio.py
print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# Hàm khởi tạo mô hình video
def init_video_models():
    global text_encoder, text_encoder_2, vae, image_encoder, transformer, tokenizer, tokenizer_2, feature_extractor
    if text_encoder is None:
        print("Loading video models...")
        text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
        tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
        tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
        feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
        image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
        
        vae.eval()
        text_encoder.eval()
        text_encoder_2.eval()
        image_encoder.eval()
        transformer.eval()

        if not high_vram:
            vae.enable_slicing()
            vae.enable_tiling()

        transformer.high_quality_fp32_output_for_inference = True
        print('transformer.high_quality_fp32_output_for_inference = True')

        transformer.to(dtype=torch.bfloat16)
        vae.to(dtype=torch.float16)
        image_encoder.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)
        text_encoder_2.to(dtype=torch.float16)

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        image_encoder.requires_grad_(False)
        transformer.requires_grad_(False)

        if not high_vram:
            DynamicSwapInstaller.install_model(transformer, device=gpu)
            DynamicSwapInstaller.install_model(text_encoder, device=gpu)
        else:
            text_encoder.to(gpu)
            text_encoder_2.to(gpu)
            image_encoder.to(gpu)
            vae.to(gpu)
            transformer.to(gpu)

# Hàm khởi tạo mô hình ảnh
def init_image_model():
    global diffuser
    if diffuser is None:
        print("Loading image model...")
        diffuser = DifussionHandler(
            use_custom=True,
            load_quantized_custom=False,
            save_quantized_model=True,
            model_type='diffusion',
            use_fp8=True,
            model_id="black-forest-labs/FLUX.1-dev",
            local_model_path='',
            lora_id='lora_weights/self/test',
            local_lora_path='/home/naver/Documents/HieuDM/manga_gen/FramePack/little-match-girl/little-match-girl.safetensors',
            use_lora=True,
            use_local_model=True,
            is_save_model=False,
            use_safetensors=True,
        )

# Hàm dọn dẹp mô hình
def cleanup_models():
    global diffuser, text_encoder, text_encoder_2, vae, image_encoder, transformer
    print("Cleaning up models...")
    if diffuser is not None:
        del diffuser
        diffuser = None
    if text_encoder is not None:
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        text_encoder = text_encoder_2 = vae = image_encoder = transformer = None
    torch.cuda.empty_cache()
    return "Models cleaned up successfully!"

# Hàm xử lý sinh video
@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return

def process_video(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    global stream
    if input_image is None:
        yield "Please upload an image!", None, None, "", "", gr.update(interactive=True), gr.update(interactive=False)
        return

    init_video_models()
    yield None, None, None, "", "Starting video generation...", gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()
    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf)

    output_filename = None
    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), "", gr.update(interactive=True), gr.update(interactive=False)
            break

def end_process():
    stream.input_queue.push('end')

# Hàm xử lý sinh ảnh
def generate_images(prompt, seeds_input, num_inference_steps, guidance_scale, image_width, image_height, save_dir):
    if not prompt.strip():
        return "Please enter a prompt!", [], None
    
    try:
        seed_list = [int(seed.strip()) for seed in seeds_input.split(",") if seed.strip()]
        if not seed_list:
            return "Please enter at least one seed!", [], None
        seed_list = seed_list[:2]  # Limit to 2 seeds for RAM efficiency
    except ValueError:
        return "Seeds must be integers!", [], None

    os.makedirs(save_dir, exist_ok=True)
    
    init_image_model()
    try:
        images = diffuser.infer_from_prompt(
            prompt=prompt,
            batch_size=len(seed_list),
            num_infer_step=num_inference_steps,
            guidance_scale=guidance_scale,
            image_shape=(image_height, image_width),
            max_sequence_length=512,
            random_cuda=True,
            cuda_idxs=seed_list,
            use_compel=False
        )
        
        pil_images = []
        file_paths = []
        for seed_idx, seed in enumerate(seed_list):
            seed_dir = os.path.join(save_dir, str(seed))
            os.makedirs(seed_dir, exist_ok=True)
            
            seed_images = images[seed_idx] if isinstance(images[seed_idx], list) else [images[seed_idx]]
            for img_idx, image in enumerate(seed_images):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                
                save_path = os.path.join(seed_dir, f'lila_{seed}_{img_idx}.png')
                try:
                    image.save(save_path)
                    print(f'[DEBUG] Saved image to: {save_path}')
                except Exception as e:
                    return f"Error saving image {img_idx} for seed {seed}: {str(e)}", [], None
                
                pil_images.append(image)
                file_paths.append(save_path)
        
        return f"Images saved at: {', '.join(file_paths)}", pil_images, file_paths
    
    except Exception as e:
        return f"Error: {str(e)}", [], None

# Hàm xử lý phân tích câu chuyện
def process_story(story_text, main_character):
    if not story_text.strip():
        return "Please enter the story text!", None
    
    if not main_character.strip():
        main_character = "Little Match Girl"
    
    try:
        scenes = extract_scenes_with_gemini(story_text, main_character)
        if not scenes:
            return "Failed to extract scenes. Check API key or try again!", None
        
        result = ""
        scene_prompts = []
        for scene in scenes:
            result += (
                f"**Scene {scene['scene_id']}**:\n"
                f"Text-to-Image Prompt: {scene['text_to_image_prompt']}\n"
                f"Image-to-Video Prompt: {scene['image_to_video_prompt']}\n\n"
            )
            scene_prompts.append({
                "scene_id": scene["scene_id"],
                "text_to_image": scene["text_to_image_prompt"],
                "image_to_video": scene["image_to_video_prompt"]
            })
        
        return result, scene_prompts
    except Exception as e:
        return f"Error: {str(e)}", None

# Hàm chọn prompt từ scene
def select_prompt(scene_id, prompt_type, scene_prompts):
    if not scene_prompts:
        return ""
    for scene in scene_prompts:
        if scene["scene_id"] == int(scene_id):
            return scene[prompt_type]
    return ""

# Tạo giao diện Gradio
css = make_progress_bar_css()
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Manga AI Generator")
    gr.Markdown("Generate comic scenes, static images, or videos from text and images.")

    with gr.Tabs():
        # Tab 1: Phân tích câu chuyện
        with gr.Tab("Story Extraction"):
            with gr.Row():
                with gr.Column():
                    story_input = gr.Textbox(
                        label="Story Text",
                        placeholder="Enter your story here...",
                        lines=10
                    )
                    character_input = gr.Textbox(
                        label="Main Character",
                        value="Little Match Girl",
                        placeholder="Enter the main character's name..."
                    )
                    story_submit = gr.Button("Extract Scenes")
                    story_cleanup = gr.Button("Cleanup Models")
                with gr.Column():
                    story_output = gr.Markdown(label="Extracted Scenes")
                    story_state = gr.State()  # Store scene prompts
            
            story_submit.click(
                fn=process_story,
                inputs=[story_input, character_input],
                outputs=[story_output, story_state]
            )
            story_cleanup.click(
                fn=cleanup_models,
                inputs=[],
                outputs=[story_output]
            )

        # Tab 2: Sinh ảnh tĩnh
        with gr.Tab("Generate Static Images"):
            with gr.Row():
                with gr.Column():
                    img_prompt = gr.Textbox(
                        label="Prompt",
                        value="A close-up of an 8-year-old Little Match Girl with long blonde curly hair, pale skin, sad eyes, in a tattered gray dress, barefoot in snow, clutching matches, under a dim streetlamp with dark Victorian buildings blurred in the background",
                        lines=5
                    )
                    img_scene_id = gr.Number(label="Scene ID (from Story Extraction)", value=1, precision=0)
                    img_load_prompt = gr.Button("Load Prompt from Scene")
                    img_seeds = gr.Textbox(
                        label="Seeds (comma-separated, max 2)",
                        value="22176",
                        placeholder="e.g., 22176, 12345"
                    )
                    img_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=30, value=20, step=1, info="Lower for faster generation")
                    img_guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=5.0, value=3.0, step=0.1)
                    img_width = gr.Slider(label="Image Width", minimum=256, maximum=512, value=512, step=8)
                    img_height = gr.Slider(label="Image Height", minimum=256, maximum=512, value=512, step=8)
                    img_save_dir = gr.Textbox(
                        label="Save Directory",
                        value="/home/naver/Documents/HieuDM/hieut/demo"
                    )
                    img_submit = gr.Button("Generate Images")
                    img_cleanup = gr.Button("Cleanup Models")
                
                with gr.Column():
                    img_output_text = gr.Markdown(label="Status")
                    img_output_gallery = gr.Gallery(label="Generated Images", columns=2, height="auto")
                    img_output_files = gr.File(label="Download Images", visible=False)
            
            img_load_prompt.click(
                fn=select_prompt,
                inputs=[img_scene_id, gr.State(value="text_to_image"), story_state],
                outputs=img_prompt
            )
            img_submit.click(
                fn=generate_images,
                inputs=[img_prompt, img_seeds, img_steps, img_guidance, img_width, img_height, img_save_dir],
                outputs=[img_output_text, img_output_gallery, img_output_files]
            )
            img_cleanup.click(
                fn=cleanup_models,
                inputs=[],
                outputs=[img_output_text]
            )

        # Tab 3: Sinh video
        with gr.Tab("Generate Video"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
                    prompt = gr.Textbox(label="Prompt", value='')
                    vid_scene_id = gr.Number(label="Scene ID (from Story Extraction)", value=1, precision=0)
                    vid_load_prompt = gr.Button("Load Prompt from Scene")
                    example_quick_prompts = gr.Dataset(
                        samples=[
                            ['The girl dances gracefully, with clear movements, full of charm.'],
                            ['A character doing some simple body movements.']
                        ],
                        label='Quick List',
                        samples_per_page=1000,
                        components=[prompt]
                    )
                    example_quick_prompts.click(
                        fn=lambda x: x[0],
                        inputs=[example_quick_prompts],
                        outputs=prompt,
                        show_progress=False,
                        queue=False
                    )

                    with gr.Row():
                        start_button = gr.Button(value="Start Generation")
                        end_button = gr.Button(value="End Generation", interactive=False)
                        vid_cleanup = gr.Button("Cleanup Models")

                    with gr.Group():
                        use_teacache = gr.Checkbox(
                            label='Use TeaCache',
                            value=True,
                            info='Faster speed, but often makes hands and fingers slightly worse.'
                        )
                        n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                        seed = gr.Number(label="Seed", value=31337, precision=0)
                        total_second_length = gr.Slider(
                            label="Total Video Length (Seconds)",
                            minimum=1,
                            maximum=120,
                            value=5,
                            step=0.1
                        )
                        latent_window_size = gr.Slider(
                            label="Latent Window Size",
                            minimum=1,
                            maximum=33,
                            value=9,
                            step=1,
                            visible=False
                        )
                        steps = gr.Slider(
                            label="Steps",
                            minimum=1,
                            maximum=100,
                            value=25,
                            step=1,
                            info='Changing this value is not recommended.'
                        )
                        cfg = gr.Slider(
                            label="CFG Scale",
                            minimum=1.0,
                            maximum=32.0,
                            value=1.0,
                            step=0.01,
                            visible=False
                        )
                        gs = gr.Slider(
                            label="Distilled CFG Scale",
                            minimum=1.0,
                            maximum=32.0,
                            value=10.0,
                            step=0.01,
                            info='Changing this value is not recommended.'
                        )
                        rs = gr.Slider(
                            label="CFG Re-Scale",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.01,
                            visible=False
                        )
                        gpu_memory_preservation = gr.Slider(
                            label="GPU Inference Preserved Memory (GB) (larger means slower)",
                            minimum=6,
                            maximum=128,
                            value=6,
                            step=0.1,
                            info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed."
                        )
                        mp4_crf = gr.Slider(
                            label="MP4 Compression",
                            minimum=0,
                            maximum=100,
                            value=16,
                            step=1,
                            info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs."
                        )
                    
                with gr.Column():
                    preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                    result_video = gr.Video(
                        label="Finished Frames",
                        autoplay=True,
                        show_share_button=False,
                        height=512,
                        loop=True
                    )
                    gr.Markdown(
                        'Note that the ending actions will be generated before the starting actions due to the inverted sampling. '
                        'If the starting action is not in the video, you just need to wait, and it will be generated later.'
                    )
                    progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                    progress_bar = gr.HTML('', elem_classes='no-generating-animation')

        vid_load_prompt.click(
            fn=select_prompt,
            inputs=[vid_scene_id, gr.State(value="image_to_video"), story_state],
            outputs=prompt
        )
        ips = [
            input_image, prompt, n_prompt, seed, total_second_length,
            latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation,
            use_teacache, mp4_crf
        ]
        start_button.click(
            fn=process_video,
            inputs=ips,
            outputs=[
                result_video, preview_image, progress_desc,
                progress_bar, start_button, end_button
            ]
        )
        end_button.click(fn=end_process)
        vid_cleanup.click(
            fn=cleanup_models,
            inputs=[],
            outputs=[progress_desc]
        )

demo.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=True
)