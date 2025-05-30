from dotenv import load_dotenv
import os

load_dotenv()

CHATGPT_KEY = os.getenv('CHATGPT_KEY')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')