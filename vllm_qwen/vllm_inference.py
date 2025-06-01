import requests
import base64
import os
import json
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import argparse
import pandas as pd
import concurrent.futures
from multiprocessing import Manager, Lock
import time

class VLMessageClient:
    def __init__(self, api_url):
        self.api_url = api_url
        self.session = requests.Session() 

    def _encode_image(self, image_path):
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def build_messages(self, item, image_root):
        image_path = os.path.join(image_root, item['images'][0])
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                    {
                        "type": "text",
                        "text": f"{item['problem']}"
                    }
                ]
            }
        ]

    def format_messages(self, messages):
        formatted = []
        for msg in messages:
            new_msg = {"role": msg["role"], "content": []}

            if msg["role"] == "system":
                new_msg["content"] = msg["content"][0]["text"]
            else:
                for part in msg["content"]:
                    if part["type"] == "image_url":
                        img_path = part["image_url"]["url"].replace("file://", "")
                        base64_image = self._encode_image(img_path)
                        new_part = {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                        new_msg["content"].append(new_part)
                    else:
                        new_msg["content"].append(part)
            formatted.append(new_msg)
        return formatted

    def process_item(self, item, image_root, output_file, total_counter, lock):
        max_retries = 3
        attempt = 0
        result = None

        while attempt < max_retries:
            try:
                attempt += 1
                raw_messages = self.build_messages(item, image_root)
                formatted_messages = self.format_messages(raw_messages)

                payload = {
                    "model": "UnifiedReward",
                    "messages": formatted_messages,
                    "temperature": 0,
                    "max_tokens": 4096,
                }

                response = self.session.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    timeout=30 + attempt*5 
                )
                response.raise_for_status()

                output = response.json()["choices"][0]["message"]["content"]
                
                with lock:
                    total_counter.value += 1

                result = {
                    "question": item["problem"],
                    "image_path": item["images"],
                    "model_output": output,
                    "attempt": attempt,
                    "success": True
                }
                break  

            except Exception as e:
                if attempt == max_retries:
                    result = {
                        "question": item["problem"],
                        "image_path": item["images"],
                        "error": str(e),
                        "attempt": attempt,
                        "success": False
                    }
                    raise(e)
                else:
                    sleep_time = min(2 ** attempt, 10)
                    time.sleep(sleep_time)

        if result:
            with lock:
                with open(output_file, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
        return result, result.get("success", False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", default="http://ip_address:8080")
    parser.add_argument("--prompt_path", default="XXXX.json")
    parser.add_argument("--image_root", default="/path/to/your/image/path")
    parser.add_argument("--output_path", default="./results.json")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of worker processes")
    args = parser.parse_args()

    with Manager() as manager:
        total_counter = manager.Value('i', 0)
        lock = manager.Lock()

    with open(args.prompt_path, "r") as f:
        test_data = json.load(f)

        # test_data = [{
        #     "prompt": "",
        #     "images": [
        #         "image_path/xxx.png"
        #     ],
        #     "problem": "",
        # },]

        open(args.output_path, "w").close()
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            client = VLMessageClient(args.api_url)

            for item in test_data:
                futures.append(
                    executor.submit(
                        client.process_item,
                        item=item,
                        image_root=args.image_root,
                        output_file=args.output_path,
                        total_counter=total_counter,
                        lock=lock
                    )
                )

            with tqdm(total=len(test_data), desc="inferencing...") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result, _ = future.result()
                        print(result)
                    except Exception as e:
                        print(f"Error: {str(e)}")
                    finally:
                        pbar.update(1)
                        current_total = total_counter.value
                        processed_info = f"{current_total}/{len(test_data)}"
                        pbar.set_postfix({
                            "processed": processed_info
                        })

        success_count = total_counter.value
        print(f"\nStatics:")
        print(f"Total data: {len(test_data)}")
        print(f"Success ratio: {success_count} ({success_count/len(test_data):.2%})")

if __name__ == "__main__":
    main()