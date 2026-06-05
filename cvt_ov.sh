#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$SCRIPT_DIR/python-env/bin/activate"
OPTIMUM_CLI="$SCRIPT_DIR/python-env/bin/optimum-cli"
# dependcy
# pip install openvino-tokenizers openvino nncf optimum[intel]
# uv pip install --index-url https://pypi.org/simple -U huggingface-hub

mkdir -p models
cd models

# model_id='Qwen/Qwen2.5-VL-7B-Instruct'
# model_id='Qwen/Qwen2.5-VL-3B-Instruct'
# model_id='Qwen/Qwen3.5-35B-A3B-Base'
# model_id='Qwen/Qwen3-Omni-30B-A3B-Instruct'
# model_id='katuni4ka/tiny-random-qwen2.5-vl/'
# model_id='katuni4ka/tiny-random-minicpmv-2_6'
# model_id='OpenVINO/Qwen2-0.5B-int8-ov'
model_id='z-lab/Qwen3-4B-DFlash-b16'
model_id='Qwen/Qwen3-4B'
# model_id='OpenVINO/phi-2-int8-ov'
model_id='openai/whisper-tiny'
model_id='OpenVINO/whisper-base-fp16-ov'
model_id='llmware/speech-t5-tts-ov'
model_id='OpenVINO/stable-diffusion-v1-5-int8-ov'
model_id='/home/xiping/mygithub/modular_genai/openvino.mx/tests/test_models/qwen3-vl-2b-instruct'
model_id='google/gemma-4-12B'

# Refer: https://hf-mirror.com/
export HF_ENDPOINT=https://hf-mirror.com
hf download --token [your token] $model_id --local-dir ./models/$model_id

# compress to int4
# optimum-cli export openvino --model $model_id --task image-text-to-text ov/qwen3-vl-2b-instruct_int4 --weight-format int4 --trust-remote-code
# optimum-cli export openvino --model $model_id --task video-text-to-text ov/Qwen2-VL-2B-Instruct_video/INT4 --weight-format int4 --trust-remote-code

# model_id='Qwen/Qwen2.5-VL-3B-Instruct'
# model_id='Qwen/Qwen2.5-VL-7B-Instruct'
# model_id='katuni4ka/tiny-random-qwen2.5-vl/'
# model_id='katuni4ka/tiny-random-qwen2vl'
# model_id='katuni4ka/tiny-random-minicpmv-2_6'
# optimum-cli export openvino --model $model_id --task image-text-to-text $model_id/INT4 --weight-format int4 --trust-remote-code
