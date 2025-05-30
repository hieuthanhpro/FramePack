import websocket, uuid, json, urllib.request, urllib.parse, io, os, shutil, random
from PIL import Image
from copy import deepcopy

class ImageGenerationController:
  def __init__(self, save_dir="/home/naver/Documents/HieuDM/PycharmProjects/manga-generation-diffusion/result_images", 
               workflow_path="/home/naver/Documents/HieuDM/PycharmProjects/manga-generation-diffusion/workflows/demo_workflow_api.json", 
               server_address="127.0.0.1:8188"):
    self.save_image_dir = save_dir
    self.base_prompt = json.load(open(workflow_path, 'r'))
    self.dir_identifier = ""

    self.output_node_index = "9"
    self.server_result_dir = "/home/naver/Documents/HieuDM/PycharmProjects/manga-generation-diffusion/ComfyUI"

    self.ws = websocket.WebSocket()
    self.server_address = server_address
    self.client_id = str(uuid.uuid4())
    try:
      self.ws.connect("ws://{}/ws?clientId={}".format(server_address, self.client_id))
    except Exception as e:
      print('Failed to create socket connection!', e)

  def queue_prompt(self, _prompt: str):
    prompt = deepcopy(self.base_prompt)
    prompt["6"]["inputs"]["text"] = _prompt

    # for generating new images everytime
    prompt["25"]["inputs"]["noise_seed"] = random.randrange(567827722483458, 976687653493075)

    data = {"prompt": prompt, "client_id": self.client_id}
    data = json.dumps(data).encode('utf-8')
    _request = urllib.request.Request("http://{}/prompt".format(self.server_address), data=data)
    return json.loads(urllib.request.urlopen(_request).read())
  
  def get_history(self, prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, prompt_id)) as response:
      return json.loads(response.read())
    
  def generate(self, _prompt: str):
    prompt_id = self.queue_prompt(_prompt)['prompt_id']
    output_images = {}
    current_node = ""
    while True:
      out = self.ws.recv()
      if isinstance(out, str):
        message = json.loads(out)
        if message['type'] == 'executing':
          data = message['data']
          if data['prompt_id'] == prompt_id:
            if data['node'] is None:
              break #Execution is done
            else:
              current_node = data['node']
        else:
          print(current_node, message["type"], message)
          if current_node == self.output_node_index and message['type'] == 'executed':
            data = message["data"]
            result_img_name = data["output"]["images"][0]["filename"]
            result_subfolder = data["output"]["images"][0]["subfolder"]
            result_output_dir = data["output"]["images"][0]["type"]
            _output_path = os.path.join(self.server_result_dir, result_output_dir, result_subfolder, result_img_name)
            shutil.copy(_output_path, f"{self.save_image_dir}/{self.dir_identifier}/{result_img_name}")

  def generate_images(self, prompts=[]):
    # self.dir_identifier = "".join(s[0] for s in prompts[0].lower().strip().split(' '))[:20]
    # if os.path.exists(f"{self.save_image_dir}/{self.dir_identifier}"):
    #   os.removedirs(f"{self.save_image_dir}/{self.dir_identifier}")
    for _prompt in prompts:
      self.generate(_prompt)

  def clear(self):
    self.dir_identifier = "otf_generation"
    self.backup_dir = "otf_backup"
    os.makedirs(f"{self.save_image_dir}/{self.dir_identifier}", exist_ok=True)
    os.makedirs(f"{self.save_image_dir}/{self.backup_dir}", exist_ok=True)
    for _file in os.listdir(f"{self.save_image_dir}/{self.dir_identifier}"):
      shutil.move(f"{self.save_image_dir}/{self.dir_identifier}/{_file}", f"{self.save_image_dir}/{self.backup_dir}/{_file}")

if __name__ == "__main__":
  generator = ImageGenerationController()
  generator.generate_images(["WamenLoraTrigger, her blonde hair styled elegantly, smiling softly as she looks directly into the camera. Her youthful beauty shines in this portrait, with her delicate features and bright eyes highlighted against a simple grey background. The neutral backdrop keeps all the focus on her radiant expression. Soft, natural lighting enhances her graceful presence, giving a magical yet natural feel to the image. Shot with a Sony Alpha 7R IV and 85mm f/1.4 lens for detailed sharpness and smooth bokeh."])
  