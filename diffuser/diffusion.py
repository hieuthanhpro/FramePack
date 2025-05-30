# importing
import os
import torch
from multiprocessing import Process, Event
from copy import deepcopy
import time

from PIL import Image
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import AutoencoderKL
from utils.images import make_img_grid, save_image
from utils.env import HUGGINGFACE_TOKEN
from utils.time import timer
from huggingface_hub import login

from optimum.quanto import freeze, qfloat8, quantize

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

from compel import Compel, ReturnedEmbeddingsType


def model_quantization(transformer, text_encoder_2, save_dir, is_loraed=False):
    # # quantize model
    t_proc = Process(target=timer, args=("QUANTIZATION",))
    passed = True
    print('Quantizing model, please hold...')
    t_proc.start()
    try:
      quantize(transformer, weights=qfloat8)
      freeze(transformer)
    except Exception as e:
      print('Failed quantizing transformer layers', e)
      passed = False
    try:
      quantize(text_encoder_2, weights=qfloat8)
      freeze(text_encoder_2)
    except Exception as e:
      print("Failed quantizing text_encoder, ", e)
      passed = False
    
    time.sleep(5)
    t_proc.terminate()
    t_proc.join()
    print('\nQuantization complete!' if passed else "\nQuantization failed!")
    if not passed:
      exit()
    # #

    # # # save the damn model
    # print('Trying to save quantized models to torch format...')
    # transformer_path = f'{save_dir}/q_transformer.pt'
    # encoder_path = f'{save_dir}/q_text_encoder_2.pt'
    # if is_loraed:
    #   transformer_path = os.path.join(transformer_path.split(os.sep)[:-1], "lora_" + os.path.basename(transformer_path))
    #   encoder_path = os.path.join(encoder_path.split(os.sep)[:-1], "lora_" + os.path.basename(encoder_path))
    # try:
    #   torch.save(transformer, transformer_path)
    #   torch.save(text_encoder_2, encoder_path)
    #   print('Saved quantized models!')
    # except Exception as e:
    #   print('Failed to save quantized model to torch', e)

    # #

def save_pipeline_torch(pipe, save_path):
    torch.save(pipe, save_path)


class LargeFluxPipeline():
    def __init__(self,
                bfl_repo='black-forest-labs/FLUX.1-dev',
                revision='refs/pr/3',
                load_quantized=False,
                load_fp8=True,
                save_quantized=False,
                fp8_transformer_id='https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors',
                q_encoder2_path='/home/naver/Documents/HieuDM/PycharmProjects/manga-generation-diffusion/models/black-forest-labs/FLUX.1-dev/q_text_encoder_2.pt',
                q_trans_path='/home/naver/Documents/HieuDM/PycharmProjects/manga-generation-diffusion/models/black-forest-labs/FLUX.1-dev/q_transformer.pt',
                dtype=torch.bfloat16,
                ) -> None:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", 
                                                    #  torch_dtype=dtype
                                                     )
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", 
                                                  # torch_dtype=dtype
                                                  )
        tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", 
                                                      # torch_dtype=dtype,
                                                      revision=revision)
        vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", 
                                            # torch_dtype=dtype, 
                                            revision=revision)
        if not load_quantized:
            if load_fp8:
                print('Loading FP8 transformer...')
                try:
                  transformer = FluxTransformer2DModel.from_single_file(fp8_transformer_id, dtype=torch.float8_e4m3fn)
                except Exception as e:
                  print(e)
                  exit()
            else:
                print('Loading normal transformer...')
                transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)
            text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
        else:
            print('Loading quantized transfomer...')
            transformer = torch.load(q_trans_path)
            text_encoder_2 = torch.load(q_encoder2_path)

        # # save quantized if needed
        if save_quantized:
            model_quantization(deepcopy(transformer), deepcopy(text_encoder_2), 
            "/home/naver/Documents/HieuDM/PycharmProjects/manga-generation-diffusion/models/black-forest-labs/FLUX.1-dev")

        pipe = None

        try:
          # create pipeline
          pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            #   text_encoder_2=None,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
          )
          pipe.text_encoder_2 = text_encoder_2
          pipe.transformer = transformer
          # # perform this in the main handler class
          # pipe.enable_model_cpu_offload()

          # memory optimization
          pipe.vae.enable_slicing()
          pipe.vae.enable_tiling()
        except Exception as e:
          print('Error creating element FluxPipeline', e)
          exit()

        self.pipe = pipe
    
    def get_pipe(self):
        if self.pipe == None:
          print('Failed to create pipeline')
          exit()
        return self.pipe
        

class DifussionHandler():
    def __init__(self,
                 use_custom=False,
                 load_quantized_custom=True,
                 save_quantized_model=False,
                 model_type='stable-diffusion',
                 model_id="stabilityai/stable-diffusion-2-1",
                 use_fp8=False,
                 load_safetensors=False,
                 lora_id="",
                 local_model_path=f'',
                 local_lora_path=f'',
                 use_lora=False,
                 use_local_model=False,
                 use_attention_slicing=True,
                 use_vae=True,
                 is_save_model=False,
                 use_safetensors=False,
                 ) -> None:
        login(token=HUGGINGFACE_TOKEN)
        print('Loading up diffuser...')
        # variable declaration
        self.use_custom = use_custom
        self.load_quantized_custom = load_quantized_custom
        self.model_type = model_type
        self.use_fp8 = use_fp8
        self.model_id = model_id    # model version
        self.lora_id = lora_id      # lora weights id
        self.local_model_path = local_model_path if local_model_path != '' else f'models/{self.model_id}'
        self.local_lora_path = local_lora_path if local_lora_path != '' else f'lora_weights/{self.lora_id}'
        self.load_safetensors = load_safetensors

        self.use_lora = use_lora
        self.use_local_model = use_local_model
        self.use_attention_slicing = use_attention_slicing
        self.use_vae = use_vae
        self.is_save_model = is_save_model
        self.use_safetensors = use_safetensors

        self.is_save_quantized = save_quantized_model

        # # book-prompter section
        self.current_char = ''
        self.current_book = ''
        self.current_iteration = ''
        # #

        # # initialization process
        self.__init_pipeline__()

        # # initialize Compel
        print('Checking before initialize Compel...')
        print(type(self.pipe.tokenizer))
        print(type(self.pipe.tokenizer_2))
        print(type(self.pipe.text_encoder))
        print(type(self.pipe.text_encoder_2))
        self.compel = Compel(
           tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
           text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
           returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
           requires_pooled=[False, True],
        )

        # # save model if specified
        if self.is_save_model: self.save_pipeline()

        print("Finished diffuser handler initialization!")

    def __init_pipeline__(self):
        # #
        if self.use_custom:
            print('Loading up custom pipeline, this might take sometime...')
            pipe = LargeFluxPipeline(
                load_quantized=self.load_quantized_custom,
                load_fp8=self.use_fp8
            )
            pipe = pipe.get_pipe()
            print('Finished loading custom pipeline!')
        elif False:
            pass
        elif not self.use_local_model:
            if self.model_type == 'stable-diffusion':
                # load huggingface model
                pipe = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=self.use_safetensors,
                )   #load with float16 for better performance
            elif self.model_type == 'diffusion':
                pipe = DiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.torch.bfloat16,
                    use_safetensors=self.use_safetensors
                )
        else:
            if not self.load_safetensors:
                if self.model_type == 'stable-diffusion':
                    # load huggingface model
                    pipe = StableDiffusionPipeline.from_pretrained(
                        self.local_model_path,
                        torch_dtype=torch.float16,
                        use_safetensors=self.use_safetensors,
                    )   #load with float16 for better performance
                elif self.model_type == 'diffusion':
                    pipe = DiffusionPipeline.from_pretrained(
                        self.local_model_path,
                        torch_dtype=torch.torch.bfloat16,
                        use_safetensors=self.use_safetensors
                    )
            else:
                if self.model_type == 'stable-diffusion':
                    # load huggingface model
                    pipe = StableDiffusionPipeline.from_single_file(
                        self.local_model_path,
                    )
                elif self.model_type == 'diffusion':
                    pipe = DiffusionPipeline.from_pretrained(
                        self.local_model_path,
                        use_safetensors=True
                    )
          
        if self.use_lora:
            print('Trying to load LORA!')
            try:
                pipe.load_lora_weights(self.local_lora_path)
                print(f'[INFO] Load lora weight "{self.lora_id}" from safetensors!')
            except Exception as e:
                print('Failed to load LORA!', e)
                exit()

        # pipe = pipe.to("cuda")
        self.pipe = pipe

        if self.use_attention_slicing:
          pipe.enable_attention_slicing() #memory optimization

        if self.is_save_quantized:
          self.quantize_pipeline()

        # if not self.use_custom:
        #     pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)   # different scheduler for dynamic stepping

        #     if self.use_vae:
        #         vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
        #         pipe.vae = vae

        #     pipe.enable_vae_tiling()
        self.pipe.enable_model_cpu_offload()
        # pipe.enable_xformers_memory_efficient_attention()

        # self.pipe = pipe
        # #

    def quantize_pipeline(self):
        if self.use_lora:
            print('Fusing lora before quantization')
            self.pipe.fuse_lora()
            # self.pipe.unload_lora_weights()
            print("Trying to perform model quantization...")
            try:
              model_quantization(self.pipe.transformer, self.pipe.text_encoder_2,
              "/home/naver/Documents/HieuDM/PycharmProjects/manga-generation-diffusion/models/Kijai/flux-fp8", True)
            except Exception as e:
              print("Failed quantizing model: ", e)
              exit()
        else:
          print("Receiving order to quantize model at handler, skipping ...")
          pass

    def save_pipeline(self):
        # # save model for easier access
        print("Saving model...")
        try:
            self.pipe.save_pretrained(self.local_model_path)
        except Exception as e:
            print('Cannot save pipeline built-in function, falling back to Pytorch, exception: ', e)
            torch.save(self.pipe, self.local_model_path + ".pt")
        #

    def infer_from_prompt(self, prompt='', 
                            batch_size=9, 
                            num_infer_step=25,
                            guidance_scale=1.2,
                            image_shape=(640,640),
                            max_sequence_length=512,
                            random_cuda=False,
                            cuda_idxs=[22176],
                            use_compel=False
        ):
        cuda_seed = 22176
        if random_cuda: cuda_seed = None
        input_dict = self.get_input(prompt, 
                                    batch_size=batch_size, 
                                    num_infer_step=num_infer_step,
                                    guidance_scale=guidance_scale,
                                    image_shape=image_shape,
                                    max_sequence_length=max_sequence_length,
                                    cuda_seed=cuda_seed,
                                    cuda_idxs=cuda_idxs,
                                    use_compel=use_compel
        )
        images = self.feed_to_pipe(input_dict).images
    
        # clearing up VRAM, while is this not being automated by the pipeline?
        torch.cuda.empty_cache()
        return images

    def infer_from_prompts(self, prompts=[], seed=0, num_infer_step=30):
        # manual input because im lazy to make another function, blame python for not having a function overridden function.
        generator = [torch.Generator("cuda").manual_seed(seed) for _ in range(len(prompts))]
        if not self.use_custom:
            input_dict = {"prompt": prompts, "generator": generator, "num_inference_step": num_infer_step}
        else:
            # ?
            input_dict = {
                "prompt": prompts,
                "guidance_scale": 1.2,
                "num_inference_steps": 50,
                "max_sequence_length": 512,
                "generator": generator     
            }
            # timestep-distiled
            input_dict = {
                "prompt": prompts,
                "width": 640,
                "height": 640,
                "guidance_scale": 0.0,
                "num_inference_steps": 5,
                "max_sequence_length": 256,
                "generator": generator     
            }
        images = self.feed_to_pipe(input_dict).images

        # clearing up VRAM, why is this not being automated by the pipeline?
        torch.cuda.empty_cache()
        return images
    
    def save_image(self, images, img_folder, img_name='demo', output_shape=(3,3)):
        images = make_img_grid(images, rows=output_shape[0], cols=output_shape[1])
        image_path = f"{img_folder}/{self.current_book}/{self.current_char}/ite{self.current_iteration}_mod{'-'.join(self.model_id.split('/'))}"
        os.makedirs(image_path, exist_ok=True)
        print(f'[DEBUG] image dir: {image_path}')
        save_image(images, image_path, img_name)
    
    """
    @Logic: this function take one prompt, create a list of {batch_size} prompts, feed to pipeline with a custom seed for each 'prompt'
    """
    def get_input(self, prompt, cuda_seed=22176, 
                  seed_step=0, 
                  batch_size=8, 
                  num_infer_step=30,
                  guidance_scale=1.2,
                  image_shape=(640,640),
                  max_sequence_length=512,
                  cuda_idxs=[],
                  use_compel=False                          
        ):
        generator = [torch.Generator("cuda").manual_seed(cuda_seed if cuda_seed else i+seed_step) for i in range(batch_size)]
        if len(cuda_idxs) > 0:
            assert len(cuda_idxs) == batch_size
            generator = [torch.Generator("cuda").manual_seed(seed) for seed in cuda_idxs]
        prompts = batch_size * [prompt]
        if not self.use_custom:
            if not use_compel:
              return {"prompt": prompts, "generator": generator, "num_inference_step": num_infer_step}
            else:
              conditioning, pooled = self.compel(prompts)
              return {
                "prompt_embeds": conditioning,
                "pooled_prompt_embeds": pooled,
                "generator": generator,
                "num_inference_step": num_infer_step
              }
        else:
            # timestep-distiled
            if not use_compel:
              return {
                  "prompt": prompts,
                  # "width": image_shape[0],
                  # "height": image_shape[1],
                  "guidance_scale": guidance_scale,
                  "num_inference_steps": num_infer_step,
                  "max_sequence_length": max_sequence_length,
                  "generator": generator     
              }
            else:
              conditioning, pooled = self.compel(prompts)
              return {
                "prompt_embeds": conditioning,
                "pooled_prompt_embeds": pooled,
                # "width": image_shape[0],
                # "height": image_shape[1],
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_infer_step,
                "max_sequence_length": max_sequence_length,
                "generator": generator 
              }
    
    def feed_to_pipe(self, input_dict={}):
        return self.pipe(**input_dict)

    def update_prompter_pref(self, char, book, ite_num):
        self.current_char = char
        self.current_book = book
        self.current_iteration = ite_num
