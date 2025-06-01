import asyncio
from openai import AsyncOpenAI
import base64
from io import BytesIO
import re 
from PIL import Image
import torch

def unifiedreward_score_sglang(device):


    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unifiedreward")
        
    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return score
    
    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "/image_1/path.jpg",
    ]
    prompts=[
        'input_prompt_1',
    ]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])

    metadata = {}
    score_fn = unifiedreward_score_sglang(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scores = score_fn(images, prompts, metadata)
    print("Scores:", scores)


if __name__ == "__main__":
    main()