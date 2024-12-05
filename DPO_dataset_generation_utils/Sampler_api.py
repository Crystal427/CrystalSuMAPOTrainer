import requests
import json
import random
import io
import base64
import math
from tqdm import tqdm
import argparse
from PIL import Image
from pathlib import Path

def calculate_adjusted_resolution(width, height, max_area, max_side=2048):
    """计算调整后的分辨率，确保是8的倍数，限制最长边不超过指定值，且总面积不超过给定值"""
    if width > max_side or height > max_side:
        scale = max_side / max(width, height)
        width = int(width * scale)
        height = int(height * scale)

    area = width * height
    if area > max_area:
        scale = math.sqrt(max_area / area)
        width = int(width * scale)
        height = int(height * scale)

    width = (width // 8) * 8
    height = (height // 8) * 8

    return width, height

def fetch_image_from_api(api_url, image_data):
    """向API发送请求并获取生成的图像"""
    try:
        response = requests.post(api_url, json=image_data, headers={"Content-Type": "application/json"})
        return response.json().get('images', [])
    except Exception as e:
        print(f"请求API时出错: {str(e)}")
        return []

def process_artist_images(artist_name, images_info, output_dir, api_url, args, data):
    """处理每个艺术家的图像生成工作"""
    artist_directory = output_dir / artist_name
    artist_directory.mkdir(exist_ok=True)

    for image_name, image_info in tqdm(images_info.items(), desc=f"处理 {artist_name}"):
        if image_info.get("generated", False):
            continue

        try:
            width, height = calculate_adjusted_resolution(image_info["width"], image_info["height"], args.max * args.max)

            image_data = {
                "prompt": image_info["tag"].replace("|||", "").strip(),
                "negative_prompt": "lowres,(bad),extra digits,2girls,bad hands,error,text,fewer,extra,missing,worst quality,jpeg artifacts,(low, old, early,mid)",
                "seed": random.randint(1, 1000000),
                "sampler_name": "DPM++ 2M",
                "scheduler": "Karras",
                "batch_size": args.batch_size,
                "steps": 28,
                "cfg_scale": 4.0,
                "width": width,
                "height": height,
                "send_images": True,
                "save_images": False,
                "override_settings": {"sd_model_checkpoint": args.model_name},
                "alwayson_scripts": {
                    "tiled vae": {
                        "args": ["true", 1024, 128, "false", "true", "true", "true"],
                    },
                }
            }

            images_base64 = fetch_image_from_api(api_url, image_data)

            base_name = Path(image_name).stem
            
            for i, base64_image in enumerate(images_base64):

                image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
                output_path = artist_directory / f"{base_name}_DPO{i+1}.webp"
                image.save(output_path, format="WEBP", quality=90)

            # 更新JSON
            image_info["generated"] = True
            
            # 保存进度
            with open(args.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"处理 {image_name} (艺术家: {artist_name}) 时出错: {str(e)}")

def generate_images_from_json(args):
    """从JSON文件中读取信息并生成图像"""
    api_url = f'http://{args.host}:{args.port}/sdapi/v1/txt2img'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(args.json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for artist_name, images_info in tqdm(data.items(), desc="处理艺术家"):
        process_artist_images(artist_name, images_info, output_dir, api_url, args, data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument("--json_path", type=str, required=True, help="数据集JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="temp", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=3, help="批量大小")
    parser.add_argument("--host", type=str, default="localhost", help="WebUI主机地址")
    parser.add_argument("--port", type=int, default=7860, help="WebUI端口")
    parser.add_argument("--max", type=int, default=1536, help="最大分辨率")

    args = parser.parse_args()
    generate_images_from_json(args)

if __name__ == "__main__":
    main()
