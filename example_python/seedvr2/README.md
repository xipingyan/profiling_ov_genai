# README

Start up Comfy UI pipeline guide.

# Script

```
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 创建并激活 Python 虚拟环境（建议 Python 3.10+）
python -m venv venv
# Windows 激活命令：
.\venv\Scripts\activate
# Linux 激活命令：
# source venv/bin/activate

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

#### Download model

前往 Hugging Face 的 Comfy-Org/SeedVR2 仓库下载以下两个核心文件：(https://huggingface.co/Comfy-Org/SeedVR2)

DiT 主模型（INT8 量化版）：

文件名：seedvr2_7b_int8_convrot.safetensors （约 8.3 GB）

存放路径：ComfyUI/models/diffusion_models/

VAE 模型：

文件名：seedvr2_ema_vae_fp16.safetensors

存放路径：ComfyUI/models/vae/

#### 启动comfy UI server
```
python main.py 
```

#### 运行

代浏览器中打开： http://127.0.0.1:8188

找到你要跑的模型的json文件，拖拽到浏览器中即可。