from safetensors.torch import load_file, save_file
import json

def merge_safetensors(config_path, output_path):
    # 读取配置文件
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    weight_map = config['weight_map']
    
    # 用于存储合并后的权重
    merged_weights = {}
    
    # 从两个文件加载权重
    file1_weights = load_file("G:/Crystal/Downloads/diffusion_pytorch_model-00001-of-00002.safetensors")
    file2_weights = load_file("G:/Crystal/Downloads/diffusion_pytorch_model-00002-of-00002.safetensors")
    
    # 根据weight_map合并权重
    for key, file_name in weight_map.items():
        if file_name == "diffusion_pytorch_model-00001-of-00002.safetensors":
            merged_weights[key] = file1_weights[key]
        else:
            merged_weights[key] = file2_weights[key]
    
    # 保存合并后的文件
    save_file(merged_weights, output_path)
    print(f"Merged model saved to {output_path}")

# 使用方法
config_path = "G:/Crystal/Downloads/diffusion_pytorch_model.safetensors.index.json"  # 配置文件路径
output_path = "G:/Crystal/Downloads/merged_model.safetensors"  # 输出文件路径

merge_safetensors(config_path, output_path)