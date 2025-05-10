import gradio as gr
from PIL import Image
import mimetypes
import torch
import os
import random
import copy
from llava.model.builder import load_pretrained_model
from llava.mm_utils import opencv_extract_frames, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
import io

# Load model only once
pretrained = "CodeGoat24/UnifiedReward-Think-7b"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)
model.eval()


def _load_video(path, num_video_frames=8, loader_fps=0.0, fps=None, frame_count=None):
    try:
        pil_imgs, _ = opencv_extract_frames(path, num_video_frames, loader_fps, fps, frame_count)
    except Exception as e:
        print(f"Error loading video {path}: {e}")
        pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * num_video_frames
    return pil_imgs


def _run_model(images, prompt):
    image_sizes = [img.size for img in images]
    image_tensor = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].cuda().bfloat16()

    question = '<image>\n' * len(images) + prompt

    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    with torch.cuda.amp.autocast():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096
        )
        text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    think_part = text_outputs[0].split('<think>')[1].split('</think>')[0] if '<think>' in text_outputs[0] else "No reasoning available."
    answer_part = text_outputs[0].split('<answer>')[1].split('</answer>')[0] if '<answer>' in text_outputs[0] else "No answer available."

    return think_part, answer_part


def infer(task_type, media1, media2, prompt, R1=None, R2=None):
    # Determine type by MIME
    def is_image(path): return mimetypes.guess_type(path)[0].startswith("image")
    def is_video(path): return mimetypes.guess_type(path)[0].startswith("video")

    media = [m for m in [media1, media2] if m is not None]

    if not media:
        return "Please upload at least one image or video.", ""

    media_paths = [m.name for m in media]

    media_types = ["image" if 'image' in task_type else "video" for p in media_paths]

    images = []

    for p, t in zip(media_paths, media_types):
        if t == "image":
            if is_image(p):
                img = Image.open(p).convert("RGB")
                images.append(img)
            else:
                return "Invalid image file."
        elif t == "video":
            if is_video(p):
                video_imgs = _load_video(p)
                images.extend(video_imgs)
            else:
                return "Invalid video file."

    for i in range(len(images)):
        images[i] = images[i].resize((512, 512))

    if task_type == "image_generation":

        question = f'Given a caption and two images generated based on this caption, please analyze in detail the two provided images. Evaluate them on various dimensions such as semantic consistency (how closely the image content aligns with the caption), aesthetics (composition, color usage, artistic expression), authenticity (realism and attention to detail), and any other factors you deem relevant. For each evaluation dimension, provide a score between 1-10 for both images (e.g., Image 1: 8/10, Image 2: 6/10) and provide a concise rationale for the score. Calculate the total score for each image by summing all dimension scores. Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. Then, in the <answer> tag, output exactly one of the following strings: \'Image 1 is better\' or \'Image 2 is better\' based on the total scores. No additional text is allowed in the <answer> section.\n\nExample output format:\n<think>\n1. Semantic consistency: Image 1 (9/10) - ...; Image 2 (7/10) - ...\n2. Aesthetics: Image 2 (8/10) - ...; Image 1 (8/10) - ...\n3. Authenticity: Image 1 (8/10) - ...; Image 2 (5/10) - ...\n[Additional dimensions if any]: Image 2 (8/10) - ...; Image 1 (6/10) - ...\nTotal score:\nImage 1: 9+8+8+6=31\nImage 2: 7+8+5+8=28\n</think>\n<answer>Image 1 is better</answer>\n\n**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the content of the images.**\n\nYour task is provided as follows:\nText Caption: [{prompt}]'

    elif task_type == "video_generation":
        question = f"Given a caption and two videos generated based on this caption, please analyze in detail the two provided videos. Evaluate them on various dimensions such as semantic consistency (how closely the video content aligns with the caption), temporal coherence (smoothness and logical flow of motion across frames), authenticity (realism and attention to detail), and any other factors you deem relevant. For each evaluation dimension, provide a score between 1-10 for both videos (e.g., Video 1: 8/10, Video 2: 6/10) and provide a concise rationale for the score. Calculate the total score for each video by summing all dimension scores. Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. Then, in the <answer> tag, output exactly one of the following strings: 'Video 1 is better' or 'Video 2 is better' based on the total scores. No additional text is allowed in the <answer> section.\n\nExample output format:\n<think>\n1. Semantic consistency: Video 1 (9/10) - ...; Video 2 (7/10) - ...\n2. Temporal coherence: Video 1 (8/10) - ...; Video 2 (6/10) - ...\n3. Authenticity: Video 1 (7/10) - ...; Video 2 (5/10) - ...\n[Additional dimensions if any]: Video 2 (8/10) - ...; Video 1 (6/10) - ...\nTotal score:\nVideo 1: 9+8+7+6=30\nVideo 2: 7+6+5+8=26\n</think>\n<answer>Video 1 is better</answer>\n**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given videos.**\n\nYour task is provided as follows:\nText Caption: [{prompt}]"

    elif task_type == "image_understanding":
        question = ("<image>\nGiven a question and a reference image, please analyze in detail the two provided answers (Answer 1 and Answer 2). " \
                    "Evaluate them based on the following three core dimensions:\n" \
                    "1. Semantic accuracy: How well the answer reflects the visual content of the image\n" \
                    "2. Correctness: Whether the answer is logically and factually correct\n" \
                    "3. Clarity: Whether the answer is clearly and fluently expressed\n" \
                    "You may also consider additional dimensions if you find them relevant (e.g., reasoning ability, attention to detail, multimodal grounding, etc.). " \
                    "For each dimension, provide a score from 1 to 10 for both answers, and briefly explain your reasoning. " \
                    "Then, compute the total score for each answer by explicitly adding the scores for all dimensions and showing the full calculation. " \
                    "Enclose your full reasoning within <think> and </think> tags. " \
                    "Then, in the <answer> tag, output exactly one of the following: 'Answer 1 is better' or 'Answer 2 is better'. No other text is allowed in the <answer> section.\n\n" \
                    "Example format:\n" \
                    "<think>\n" \
                    "1. Semantic accuracy: Answer 1 (9/10) - ...; Answer 2 (7/10) - ...\n" \
                    "2. Correctness: Answer 1 (8/10) - ...; Answer 2 (7/10) - ...\n" \
                    "3. Clarity: Answer 1 (9/10) - ...; Answer 2 (8/10) - ...\n" \
                    "[Additional dimensions if any]: Answer 1 (6/10) - ...; Answer 2 (7/10) - ...\n" \
                    "Total score:\nAnswer 1: 9+8+9+6=32\nAnswer 2: 7+7+8+7=29\n" \
                    "</think>\n" \
                    "<answer>Answer 1 is better</answer>\n\n" \
                    "**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given answers.**\n\n" \
                    f"Your task is provided as follows:\nQuestion: [{prompt}]\nAnswer 1: [{R1}]\nAnswer 2: [{R2}]")

    elif task_type == "video_understanding":

        question = ("Given a question and a reference video, please evaluate the two provided answers (Answer 1 and Answer 2). " \
                    "Judge them based on the following key dimensions:\n" \
                    "1. Semantic accuracy: Does the answer align with the visual and temporal content in the video?\n" \
                    "2. Correctness: Is the answer factually and logically correct?\n" \
                    "3. Clarity: Is the answer expressed fluently, clearly, and coherently?\n" \
                    "You are encouraged to consider any additional dimensions if relevant (e.g., temporal reasoning, causal understanding, visual detail, emotional perception, etc.). " \
                    "For each dimension, assign a score from 1 to 10 for both answers and explain briefly. " \
                    "Then, compute and explicitly show the total score as an addition of all dimension scores. " \
                    "Wrap your full reasoning in <think> tags. In the <answer> tag, output exactly one of the following: 'Answer 1 is better' or 'Answer 2 is better'. No additional commentary is allowed in the <answer> section.\n\n" \
                    "Example format:\n" \
                    "<think>\n" \
                    "1. Semantic accuracy: Answer 1 (8/10) - ...; Answer 2 (9/10) - ...\n" \
                    "2. Correctness: Answer 1 (7/10) - ...; Answer 2 (6/10) - ...\n" \
                    "3. Clarity: Answer 1 (9/10) - ...; Answer 2 (8/10) - ...\n" \
                    "[Additional dimensions if any]: Answer 1 (7/10) - ...; Answer 2 (6/10) - ...\n" \
                    "Total score:\nAnswer 1: 8+7+9+7=31\nAnswer 2: 9+6+8+6=29\n" \
                    "</think>\n" \
                    "<answer>Answer 1 is better</answer>\n\n" \
                    "**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given answers.**\n\n" \
                    f"Your task is provided as follows:\nQuestion: {prompt}\nAnswer 1: {R1}\nAnswer 2: {R2}\n"
    )
    # For understanding tasks, we also need answer1 and answer2
    return _run_model(images, question)
    # Run the model and return the result


# 用于更新界面元素可见性的函数
def update_visibility(task):
    if task in ["image_understanding", "video_understanding"]:
        return gr.File.update(visible=True, value=None), gr.File.update(visible=False, value=None), gr.Textbox.update(visible=True, value=""), gr.Textbox.update(visible=True, value=""), gr.Textbox(label="Question", value=""), media1_preview.update(value=None, visible=False), media2_preview.update(value=None, visible=False), media3_preview.update(value=None, visible=False), media4_preview.update(value=None, visible=False), gr.Textbox(label="Thinking Process", interactive=False, value=""), gr.Textbox(label="Answer Result", interactive=False, value="")
    else:
        return gr.File.update(visible=True, value=None), gr.File.update(visible=True, value=None), gr.Textbox.update(visible=False, value=""), gr.Textbox.update(visible=False, value=""), gr.Textbox(label="Caption", value=""), media1_preview.update(value=None, visible=False), media2_preview.update(value=None, visible=False), media3_preview.update(value=None, visible=False), media4_preview.update(value=None, visible=False), gr.Textbox(label="Thinking Process", interactive=False, value=""), gr.Textbox(label="Answer Result", interactive=False, value="")


# Build Gradio interface using gr.Blocks
with gr.Blocks(title="UnifiedReward-Think") as demo:
    gr.Markdown("UnifiedReward-Think is the first unified multimodal CoT-based reward model, capable of multi-dimensional, step-by-step long-chain reasoning for both visual understanding and generation reward tasks.")

    with gr.Row():
        task_type = gr.Dropdown(
            label="Task Selection",
            choices=["image_generation", "video_generation", "image_understanding", "video_understanding"],
            value="image_generation"
        )
        file1 = gr.File(label="Upload Media 1 (image or video)", file_types=[".jpg", ".png", ".mp4", ".mov"])
        file2 = gr.File(label="Upload Media 2 (image or video)", file_types=[".jpg", ".png", ".mp4", ".mov"])

    caption = gr.Textbox(label="Caption")
    answer1 = gr.Textbox(label="Answer 1", visible=False)
    answer2 = gr.Textbox(label="Answer 2", visible=False)
    run_button = gr.Button("Run")

    with gr.Row():
        media1_preview = gr.Image(label="Image 1 Preview", visible=False)
        media2_preview = gr.Image(label="Image 2 Preview", visible=False)

    with gr.Row():
        media3_preview = gr.Video(label="Video 1 Preview", visible=False)
        media4_preview = gr.Video(label="Video 2 Preview", visible=False)


    with gr.Row():
        output_think = gr.Textbox(label="Thinking Process", interactive=False)
        output_answer = gr.Textbox(label="Final Answer", interactive=False)

    def update_preview_1(task_type, media):
        media1_preview_visible = False
        media3_preview_visible = False
        media1_preview_value = None
        media3_preview_value = None

        if 'image' in task_type:
            media1_preview_visible = True
            media1_preview_value = media.name
        elif 'video' in task_type:
            media3_preview_visible = True
            media3_preview_value = media.name
        

        
        return media1_preview.update(value=media1_preview_value, visible=media1_preview_visible), media3_preview.update(value=media3_preview_value, visible=media3_preview_visible)


    def update_preview_2(task_type, media):
        media2_preview_visible = False
        media4_preview_visible = False
        media2_preview_value = None
        media4_preview_value = None

        if 'image' in task_type:
            media2_preview_visible = True
            media2_preview_value = media.name
        elif 'video' in task_type:
            media4_preview_visible = True
            media4_preview_value = media.name
        

        return media2_preview.update(value=media2_preview_value, visible=media2_preview_visible), media4_preview.update(value=media4_preview_value, visible=media4_preview_visible)


    file1.upload(update_preview_1, inputs=[task_type, file1], outputs=[media1_preview, media3_preview])
    file2.upload(update_preview_2, inputs=[task_type, file2], outputs=[media2_preview, media4_preview])

    task_type.change(
        fn=update_visibility,
        inputs=[task_type],
        outputs=[file1, file2, answer1, answer2, caption, media1_preview, media2_preview, media3_preview, media4_preview, output_think, output_answer]
    )

    run_button.click(
        fn=infer,
        inputs=[task_type, file1, file2, caption, answer1, answer2],
        outputs=[output_think, output_answer]
    )


if __name__ == "__main__":
    demo.launch()
