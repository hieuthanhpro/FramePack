import json
import os
import google.generativeai as genai
from time import sleep

def extract_scenes_with_gemini(story_text, main_character="Little Match Girl"):
    """
    Phân tích câu chuyện thành các cảnh truyện tranh, tạo prompt cho text-to-image và image-to-video.
    
    Args:
        story_text (str): Văn bản câu chuyện.
        main_character (str): Tên nhân vật chính (mặc định: Little Match Girl).
    
    Returns:
        list: Danh sách các cảnh dạng JSON [{'scene_id': int, 'text_to_image_prompt': str, 'image_to_video_prompt': str}].
    """
    # Thiết lập API Gemini
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCT1P_5lYmqfbmcgcT89WIs9nZ_XIKEIlk")  # Fallback nếu không có env
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    system_prompt = (
        f"Divide the story below into 4-6 comic-style scenes. For each scene with the main character '{main_character}':\n"
        f"1. Create a 'text_to_image_prompt' (50-70 words, ~70 tokens) describing the character's appearance, actions, emotions, and scenery in a concise, vivid way, suitable for a text-to-image model like FLUX.1-dev.\n"
        f"2. Create an 'image_to_video_prompt' (<20 words, ~15 tokens) describing only the character's motion for a short video clip, in one sentence.\n"
        f"Return the result as a JSON list: [{{\"scene_id\": 1, \"text_to_image_prompt\": \"...\", \"image_to_video_prompt\": \"...\"}}]."
    )

    for attempt in range(3):  # Thử lại 3 lần nếu lỗi
        try:
            response = model.generate_content([system_prompt, story_text])
            cleaned = response.text.strip().replace("```json", "").replace("```", "").strip()
            scenes = json.loads(cleaned)
            # Kiểm tra format
            for scene in scenes:
                if not all(key in scene for key in ["scene_id", "text_to_image_prompt", "image_to_video_prompt"]):
                    raise ValueError(f"Invalid scene format: {scene}")
            return scenes
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            sleep(1)
    
    # Fallback nếu thất bại
    return [
        {
            "scene_id": 1,
            "text_to_image_prompt": f"{main_character} stands in a snowy street, looking sad.",
            "image_to_video_prompt": f"{main_character} walks slowly through snow."
        }
    ]