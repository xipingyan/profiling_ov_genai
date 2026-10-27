# qwen3-omni-4b private

This is private model.

weights: Qwen3-Omni-4B-Instruct-multilingual
codes:   qwen3_omni_4B_final_release.tar

```
tar -xf qwen3_omni_4B_final_release.tar

python3 -m venv python-env
source python-env/bin/activate

cd qwen3_omni_4B_final_release

# 这2个install gpu不需要
pip install -r requirements.txt
pip install huggingface_hub

uv pip install soundfile audioread av librosa
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install accelerate==1.6.0
pip install ./transformers-internal-q3o-dense-3d1a4f5e34753e51cb85052539c6ef10cab9a5c1/
```

#### GPU

更改模型路径和测试图片，音频，在run_qwen3_omni_dense.py 中。

```
python run_qwen3_omni_dense.py 
```

#### CPU

For device cpu， please update codes of run_qwen3_omni_dense.py:

```
MODEL_PATH = "../Qwen3-Omni-4B-Instruct-multilingual"
DEVICE='cpu'

model = Qwen3OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    # device_map="auto",
    # attn_implementation="flash_attention_2",
    device_map=DEVICE,
    attn_implementation="eager"    
)
```

#### Run

```
python run_qwen3_omni_dense.py
```