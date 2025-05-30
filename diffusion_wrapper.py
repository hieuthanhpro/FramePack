# diffusion_wrapper.py
import os
import uuid
from diffuser.diffusion import DifussionHandler  # Giả định bạn đang dùng như đã gửi trước đó

class DiffusionWrapper:
    def __init__(self, save_dir="./outputs/images"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.diffuser = DifussionHandler(
            use_custom=True,
            load_quantized_custom=False,
            save_quantized_model=False,
            model_type='diffusion',
            use_fp8=True,
            model_id="black-forest-labs/FLUX.1-dev",
            local_model_path='',
            lora_id='lora_weights/self/test',
            local_lora_path='lora_weights/self/test/little-girl-v1.safetensors',
            use_lora=True,
            use_local_model=True,
            is_save_model=False,
            use_safetensors=True,
        )

    def generate_and_save_image(self, prompt: str) -> str:
        seed = [uuid.uuid4().int % 1000000000]  # Sinh seed ngẫu nhiên
        img_name = f"img_{seed[0]}"
        img_folder = os.path.join(self.save_dir, str(seed[0]))
        os.makedirs(img_folder, exist_ok=True)

        images = self.diffuser.infer_from_prompt(
            prompt=prompt,
            batch_size=1,
            num_infer_step=50,
            guidance_scale=3.0,
            image_shape=(640, 640),
            max_sequence_length=512,
            random_cuda=True,
            cuda_idxs=seed,
            use_compel=False
        )

        self.diffuser.save_image(images, img_folder, img_name)
        return os.path.join(img_folder, f"{img_name}.png")
