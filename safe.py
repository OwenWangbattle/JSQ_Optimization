from safetensors import safe_open

# 设置你的 safetensor 文件路径
file_path = "/root/autodl-tmp/results_2/model-00002-of-00002.safetensors"

import argparse
import os
from safetensors import safe_open

def inspect_safetensors(file_path):
    # 获取 safetensor 文件的大小（字节）
    file_size = os.path.getsize(file_path)
    print(f"Total File Size: {file_size} bytes")
    
    # 打开 safetensor 文件
    with safe_open(file_path, framework="pt") as f:
        # 获取文件的元数据
        metadata = f.metadata()
        print(f"Metadata: {metadata}")
        
        # 打印所有 tensor 的键（如果是 dict-like）
        print("Tensor Keys:")
        for key in f.keys():
            print(f"Tensor Name: {key}")
            
            # 获取 tensor 的具体数据
            tensor = f.get_tensor(key)  # 使用 get_tensor 方法
            dtype = tensor.dtype
            shape = tensor.shape
            device = tensor.device
            
            # 计算大小（元素个数 * 每个元素的字节数）
            num_elements = tensor.numel()
            element_size = tensor.element_size()  # 每个元素的字节数
            tensor_size = num_elements * element_size  # 总字节数
            
            # 打印详细信息
            print(f"Data Type: {dtype}")

def main():
    inspect_safetensors(file_path)

if __name__ == "__main__":
    main()
