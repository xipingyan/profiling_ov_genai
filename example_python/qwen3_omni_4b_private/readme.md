# qwen3-omni-4b private

This is private model.

weights: Qwen3-Omni-4B-Instruct-multilingual
codes:   qwen3_omni_4B_final_release.tar

```
tar -xf qwen3_omni_4B_final_release.tar

python3 -m venv python-env
source python-env/bin/activate

cd qwen3_omni_4B_final_release
pip install -r requirements.txt
pip install huggingface_hub
pip install soundfile audioread av librosa torch torchvision 
pip install ./transformers-internal-q3o-dense-3d1a4f5e34753e51cb85052539c6ef10cab9a5c1/
```

#### CPU

# MODEL_PATH = "/models/Qwen3-Omni-4B-Instruct-multilingual"
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