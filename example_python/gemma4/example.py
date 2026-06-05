from transformers import AutoProcessor, AutoModelForMultimodalLM
from PIL import Image
import torch

import random
import numpy as np
# 0. 固定全局随机种子（双重保险，确保所有底层操作可复现）
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# MODEL_ID = "google/gemma-4-12B-it"
MODEL_ID = "/home/xiping/mygithub/profiling_ov_genai/models/models/google/gemma-4-12B"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    # dtype="auto", 
    # device_map="auto"
    torch_dtype=torch.bfloat16,   # 1. 显式指定 torch.bfloat16，不要用 "auto"
    # low_cpu_mem_usage=True,       # 2. 极其重要：防止加载时内存翻倍爆掉
    device_map="cpu"              # 3. 明确指定设备为纯 CPU
)

# Prompt - add image before text
messages = [
    {
        "role": "user", "content": [
            # {"type": "image", "url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/Demos/sample-data/GoldenGate.png"},
            # local image file
            {"type": "image", "url": "./cat_120_100.png"},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

# 2. 使用 PIL 打开你的本地图片
image_path = "./cat_120_100.png"
image = Image.open(image_path).convert("RGB")

# 3. 【核心修改点】使用 Gemma 4 标准图像 Token: <|image|>
prompt = "<|im_start|>user\n<|image|>\nWhat is shown in this image?<|im_end|>\n<|im_start|>assistant\n"

# 4. 直接使用 processor 编码
print("正在处理图像和文本输入...")
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

# 5. 生成推理
print("开始生成回复（CPU 推理较慢，请耐心等待）...")
outputs = model.generate(**inputs, max_new_tokens=4)

# 6. 解码并打印输出
response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
print("模型回复: ", response)
