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

# Đăng nhập Hugging Face với token cố định
login(token="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")

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

# Cấu hình cho sinh video
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 16  # Giảm ngưỡng vì RAM 24 GB và VRAM có thể hạn chế
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
            local_lora_path='/home/naver/Documents/HieuDM/PycharmProjects/manga-generation-diffusion/lora_weights/self/test/little-girl-v1.safetensors',
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

# Hàm xử lý sinh video
@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    job_id = generate_timestamp()
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

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

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
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
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation / 1000)

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
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30).'
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
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
    stream.output_queue.push(('end', None))
    return

def process_video(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    global stream
    if input_image is None:
        yield "Vui lòng tải lên hình ảnh!", None, None, "", "", gr.update(interactive=True), gr.update(interactive=False)
        return

    init_video_models()  # Tải mô hình video
    yield None, None, None, "", "Starting video generation...", gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()
    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf)

    output_filename = None
    while True:
        flag, data = stream.output_queue.next()
        if flag == 'file':
            output_filename = data
            yield output_filename, None, None, "", "", gr.update(interactive=True), gr.update(interactive=False)
        elif flag == 'progress':
            preview, desc, html = data
            yield None, preview, None, desc, html, gr.update(interactive=False), gr.update(interactive=True)
        elif flag == 'end':
            cleanup_models()  # Dọn dẹp sau khi hoàn tất
            yield output_filename, None, None, "", "Video generation completed!", gr.update(interactive=True), gr.update(interactive=False)
            break

def end_process():
    stream.input_queue.push('end')

# Hàm xử lý sinh ảnh
def generate_images(prompt, seeds_input, num_inference_steps, guidance_scale, image_width, image_height, save_dir):
    if not prompt.strip():
        return "Vui lòng nhập prompt!", [], None
    
    try:
        seed_list = [int(seed.strip()) for seed in seeds_input.split(",") if seed.strip()]
        if not seed_list:
            return "Vui lòng nhập ít nhất một seed!", [], None
        seed_list = seed_list[:2]  # Giới hạn tối đa 2 ảnh để tiết kiệm RAM
    except ValueError:
        return "Seeds phải là số nguyên!", [], None

    os.makedirs(save_dir, exist_ok=True)
    
    init_image_model()  # Tải mô hình ảnh
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
        for idx, image in enumerate(images):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            pil_images.append(image)
            save_path = f"{save_dir}/{seed_list[idx]}/lila_{seed_list[idx]}.png"
            diffuser.save_image(
                images=[image],
                img_folder=f"{save_dir}/{seed_list[idx]}",
                img_name=f"lila_{seed_list[idx]}"
            )
            file_paths.append(save_path)
        
        cleanup_models()  # Dọn dẹp sau khi hoàn tất
        return f"Ảnh đã được lưu tại: {', '.join(file_paths)}", pil_images, file_paths
    
    except Exception as e:
        cleanup_models()
        return f"Đã xảy ra lỗi: {str(e)}", [], None

# Hàm xử lý phân tích câu chuyện
def process_story(story_text, main_character):
    if not story_text.strip():
        return "Vui lòng nhập văn bản câu chuyện!"
    if not main_character.strip():
        main_character = "Little Match Girl"
    
    try:
        scenes = extract_scenes_with_gemini(story_text, main_character)
        result = ""
        for scene in scenes:
            result += f"**Cảnh {scene['scene_number']}**:\n{scene['prompt']}\n\n"
        return result
    except Exception as e:
        return f"Đã xảy ra lỗi: {str(e)}"

# Tạo giao diện Gradio
css = make_progress_bar_css()
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Manga AI Generator")
    gr.Markdown("Tạo cảnh truyện tranh, ảnh tĩnh, hoặc video từ văn bản và hình ảnh.")

    with gr.Tabs():
        # Tab 1: Phân tích câu chuyện
        with gr.Tab("Story Extraction"):
            with gr.Row():
                with gr.Column():
                    story_input = gr.Textbox(
                        label="Văn bản câu chuyện",
                        placeholder="Nhập câu chuyện tại đây...",
                        lines=10
                    )
                    character_input = gr.Textbox(
                        label="Nhân vật chính",
                        value="Little Match Girl",
                        placeholder="Nhập tên nhân vật chính..."
                    )
                    story_submit = gr.Button("Phân tích")
                with gr.Column():
                    story_output = gr.Markdown(label="Kết quả phân tích")
            
            story_submit.click(
                fn=process_story,
                inputs=[story_input, character_input],
                outputs=story_output
            )

        # Tab 2: Sinh ảnh tĩnh
        with gr.Tab("Generate Static Images"):
            with gr.Row():
                with gr.Column():
                    img_prompt = gr.Textbox(
                        label="Prompt",
                        value="A close-up of an 8-year-old Little Match Girl with long blonde curly hair, pale skin...",
                        lines=5
                    )
                    img_seeds = gr.Textbox(
                        label="Seeds (cách nhau bằng dấu phẩy, tối đa 2)",
                        value="22176",
                        placeholder="Nhập các seed, ví dụ: 22176, 12345"
                    )
                    img_steps = gr.Slider(label="Số bước suy luận", minimum=10, maximum=50, value=30, step=1)
                    img_guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=7.0, value=3.0, step=0.1)
                    img_width = gr.Slider(label="Chiều rộng ảnh", minimum=256, maximum=640, value=512, step=8)
                    img_height = gr.Slider(label="Chiều cao ảnh", minimum=256, maximum=640, value=512, step=8)
                    img_save_dir = gr.Textbox(
                        label="Thư mục lưu ảnh",
                        value="/home/naver/Documents/HieuDM/hieut/demo"
                    )
                    img_submit = gr.Button("Sinh ảnh")
                
                with gr.Column():
                    img_output_text = gr.Markdown(label="Thông báo")
                    img_output_gallery = gr.Gallery(label="Ảnh được sinh ra", columns=2, height="auto")
                    img_output_files = gr.File(label="Tải xuống ảnh", visible=False)
            
            img_submit.click(
                fn=generate_images,
                inputs=[img_prompt, img_seeds, img_steps, img_guidance, img_width, img_height, img_save_dir],
                outputs=[img_output_text, img_output_gallery, img_output_files]
            )

        # Tab 3: Sinh video
        with gr.Tab("Generate Video"):
            with gr.Row():
                with gr.Column():
                    vid_image = gr.Image(sources='upload', type="numpy", label="Hình ảnh đầu vào", height=320)
                    vid_prompt = gr.Textbox(label="Prompt", value='')
                    quick_prompts = [['The girl dances gracefully, with clear movements, full of charm.'], ['A character doing some simple body movements.']]
                    vid_examples = gr.Dataset(samples=quick_prompts, label="Prompt mẫu", components=[vid_prompt])
                    vid_examples.click(lambda x: x[0], inputs=[vid_examples], outputs=vid_prompt, queue=False)
                    
                    with gr.Row():
                        vid_start = gr.Button("Bắt đầu sinh video")
                        vid_end = gr.Button("Dừng sinh video", interactive=False)
                    
                    with gr.Group():
                        vid_teacache = gr.Checkbox(label="Sử dụng TeaCache", value=True, info="Tăng tốc nhưng có thể giảm chất lượng tay/chân")
                        vid_n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                        vid_seed = gr.Number(label="Seed", value=31337, precision=0)
                        vid_length = gr.Slider(label="Độ dài video (giây)", minimum=1, maximum=10, value=5, step=0.1, info="Giới hạn tối đa 10 giây để tiết kiệm tài nguyên")
                        vid_latent_size = gr.Slider(label="Kích thước cửa sổ latent", minimum=1, maximum=33, value=9, visible=False)
                        vid_steps = gr.Slider(label="Số bước", minimum=10, maximum=50, value=25, step=1, info="Khuyến nghị giữ 25")
                        vid_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, visible=False)
                        vid_gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0)
                        vid_rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, visible=False)
                        vid_gpu_mem = gr.Slider(label="Bộ nhớ GPU dự trữ (MB)", minimum=2000, maximum=12000, value=8000, step=100, info="Tăng nếu gặp lỗi OOM")
                        vid_mp4_crf = gr.Slider(label="Nén MP4", minimum=0, maximum=100, value=16, info="Thấp hơn = chất lượng cao hơn")
                
                with gr.Column():
                    vid_output_video = gr.Video(label="Video được sinh ra", autoplay=True, height=512)
                    vid_preview = gr.Image(label="Khung hình xem trước", height=200, visible=False)
                    vid_progress_desc = gr.Markdown()
                    vid_progress_bar = gr.HTML('')
            
            vid_inputs = [vid_image, vid_prompt, vid_n_prompt, vid_seed, vid_length, vid_latent_size, vid_steps, vid_cfg, vid_gs, vid_rs, vid_gpu_mem, vid_teacache, vid_mp4_crf]
            vid_start.click(
                fn=process_video,
                inputs=vid_inputs,
                outputs=[vid_output_video, vid_preview, vid_progress_desc, vid_progress_bar, vid_start, vid_end]
            )
            vid_end.click(fn=end_process)

# Khởi chạy giao diện
demo.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=True
)