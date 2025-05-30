import gradio as gr
from extractor.scene_extractor import extract_scenes_with_gemini

# Hàm xử lý chính
def process_story(story_text, main_character):
    if not story_text.strip():
        return "Vui lòng nhập văn bản câu chuyện."
    if not main_character.strip():
        main_character = "Little Match Girl"  # Giá trị mặc định
    
    try:
        # Gọi hàm từ extractor.py
        scenes = extract_scenes_with_gemini(story_text, main_character)
        
        # Định dạng kết quả
        result = ""
        for scene in scenes:
            result += f"**Cảnh {scene['scene_number']}**:\n{scene['prompt']}\n\n"
        return result
    except Exception as e:
        return f"Đã xảy ra lỗi: {str(e)}"

# Tạo giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Phân Tích Câu Chuyện Thành Cảnh Truyện Tranh")
    gr.Markdown("Nhập văn bản câu chuyện và tên nhân vật chính để chia thành các cảnh truyện tranh với mô tả chi tiết.")
    
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
            submit_button = gr.Button("Phân tích")
        
        with gr.Column():
            output = gr.Markdown(label="Kết quả phân tích")
    
    # Liên kết nút với hàm xử lý
    submit_button.click(
        fn=process_story,
        inputs=[story_input, character_input],
        outputs=output
    )

# Khởi chạy giao diện
demo.launch()