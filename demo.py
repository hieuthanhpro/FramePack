import os
import gradio as gr
from diffuser.diffusion import DifussionHandler
from PIL import Image
import matplotlib.pyplot as plt
import io
import numpy as np

# Khởi tạo DifussionHandler
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

# Hàm hiển thị ảnh (chuyển đổi để tương thích với Gradio)
def display_images_for_gradio(images, titles=None):
    """
    Chuyển danh sách ảnh thành định dạng Gradio và trả về danh sách ảnh PIL.
    """
    pil_images = []
    for image in images:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        pil_images.append(image)
    return pil_images

# Hàm xử lý chính cho Gradio
def generate_images(prompt, seeds, num_inference_steps, guidance_scale, image_width, image_height, save_dir):
    if not prompt.strip():
        return "Vui lòng nhập prompt!", []
    
    # Chuyển seeds thành danh sách số nguyên
    try:
        seed_list = [int(seed.strip()) for seed in seeds.split(",") if seed.strip()]
        if not seed_list:
            return "Vui lòng nhập ít nhất một seed!", []
    except ValueError:
        return "Seeds phải là số nguyên!", []
    
    # Tạo thư mục lưu ảnh
    os.makedirs(save_dir, exist_ok=True)
    
    # Sinh ảnh
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
        
        # Lưu ảnh
        for idx, image in enumerate(images):
            diffuser.save_image(
                images=[image],
                img_folder=f"{save_dir}/{seed_list[idx]}",
                img_name=f"lila_{seed_list[idx]}"
            )
        
        # Chuyển đổi ảnh để hiển thị trên Gradio
        pil_images = display_images_for_gradio(images)
        titles = [f"Seed: {seed}" for seed in seed_list]
        
        return "Ảnh đã được sinh và lưu thành công!", pil_images
    
    except Exception as e:
        return f"Đã xảy ra lỗi: {str(e)}", []

# Tạo giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Sinh Ảnh Manga với Diffusion Model")
    gr.Markdown("Nhập prompt, seed, và các tham số để sinh ảnh. Ảnh sẽ được lưu vào thư mục được chỉ định.")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                value="A close-up of an 8-year-old Little Match Girl with long blonde curly hair, pale skin, and sad, downcast eyes. She wears a tattered gray dress, her bare feet red and blue from the cold, clutching matches in trembling hands. Snowflakes fall around her under a dim streetlamp, with dark Victorian buildings blurred in the snowy background",
                lines=5
            )
            seeds_input = gr.Textbox(
                label="Seeds (cách nhau bằng dấu phẩy)",
                value="22176",
                placeholder="Nhập các seed, ví dụ: 22176, 12345"
            )
            num_inference_steps = gr.Slider(
                label="Số bước suy luận",
                minimum=10,
                maximum=100,
                value=50,
                step=1
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=1.0,
                maximum=10.0,
                value=3.0,
                step=0.1
            )
            image_width = gr.Slider(
                label="Chiều rộng ảnh",
                minimum=256,
                maximum=1024,
                value=640,
                step=8
            )
            image_height = gr.Slider(
                label="Chiều cao ảnh",
                minimum=256,
                maximum=1024,
                value=640,
                step=8
            )
            save_dir_input = gr.Textbox(
                label="Thư mục lưu ảnh",
                value="/home/naver/Documents/HieuDM/hieut/demo",
                placeholder="Nhập đường dẫn thư mục"
            )
            submit_button = gr.Button("Sinh ảnh")
        
        with gr.Column():
            output_text = gr.Markdown(label="Thông báo")
            output_images = gr.Gallery(
                label="Ảnh được sinh ra",
                elem_id="gallery",
                columns=3,
                object_fit="contain",
                height="auto"
            )
    
    # Liên kết nút với hàm xử lý
    submit_button.click(
        fn=generate_images,
        inputs=[
            prompt_input,
            seeds_input,
            num_inference_steps,
            guidance_scale,
            image_width,
            image_height,
            save_dir_input
        ],
        outputs=[output_text, output_images]
    )

# Khởi chạy giao diện
demo.launch()