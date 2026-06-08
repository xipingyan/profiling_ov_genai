from transformers import AutoProcessor, AutoModelForMultimodalLM
from PIL import Image
import torch
# pip install librosa

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
MODEL_ID = "/home/xiping/mygithub/profiling_ov_genai/models/models/google/gemma-4-12B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 2. 使用 PIL 打开你的本地图片
image_path = "./cat_120_100.png"
image_path = './GoldenGate.png'
# image = Image.open(image_path).convert("RGB")

# # 3. 【核心修改点】使用 Gemma 4 标准图像 Token: <|image|>
# prompt = "<|im_start|>user\n<|image|>\nWhat is shown in this image?<|im_end|>\n<|im_start|>assistant\n"
# # prompt = "What is shown in this image?"

# # 4. 直接使用 processor 编码
# print("正在处理图像和文本输入...")
# # inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
# inputs = processor(text=prompt, images=image, return_tensors="pt")
# input_len = inputs["input_ids"].shape[-1]

# Prompt - add image before text
messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": image_path},
            {"type": "text", "text": "What is shown in this image?"},
            # {"type": "audio", "audio": "journal1.wav"},
            # {"type": "text", "text": "Transcribe the following speech segment."},
        ]
    }
]
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Write a short joke about saving RAM."},
# ]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
)
input_len = inputs["input_ids"].shape[-1]


def run_torch():
    """原始 Torch 推理路径"""
    print("\n" + "=" * 60)
    print("== [Torch] 原始推理")
    print("=" * 60)

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )

    # 5. 生成推理
    print("开始生成回复（CPU 推理较慢，请耐心等待）...")
    outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    # 6. 解码并打印输出
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    print("模型回复: ", response)
    return response


def run_torch_pipeline():
    """用 Torch 逐步执行与 OV 相同的 pipeline，用于逐模块对比验证"""
    import time

    print("\n" + "=" * 60)
    print("== [Torch Pipeline] 逐步推理（与 OV 对比）")
    print("=" * 60)

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model.eval()

    input_ids = inputs["input_ids"]
    mm_token_type_ids = inputs["mm_token_type_ids"]
    mm_ids = mm_token_type_ids[0]

    # Step 1: text embeddings
    print("Step 1: embed_tokens...")
    t1 = time.time()
    with torch.no_grad():
        text_embeds = model.model.language_model.embed_tokens(input_ids).float().numpy()
    print(f"  text_embeds shape: {text_embeds.shape}, 耗时: {(time.time()-t1)*1000:.1f} ms")

    merged_embeds = text_embeds.copy()

    # Step 2: vision embeddings
    image_mask = (mm_ids == 1).numpy()
    if image_mask.any() and "pixel_values" in inputs:
        print("Step 2: embed_vision...")
        pixel_values = inputs["pixel_values"]
        image_position_ids = inputs["image_position_ids"]
        t1 = time.time()
        with torch.no_grad():
            vision_all = model.model.embed_vision(
                pixel_values.to(torch.bfloat16), image_position_ids
            ).float().numpy()
        padding_mask = (image_position_ids.numpy() == -1).all(axis=-1)
        vision_valid = vision_all[~padding_mask]
        print(f"  vision_embeds shape: {vision_valid.shape}, 耗时: {(time.time()-t1)*1000:.1f} ms")
        merged_embeds[0, image_mask, :] = vision_valid

    # Step 3: audio embeddings
    audio_mask = (mm_ids == 3).numpy()
    if audio_mask.any() and "input_features" in inputs:
        print("Step 3: embed_audio...")
        input_features = inputs["input_features"]
        input_features_mask = inputs["input_features_mask"]
        t1 = time.time()
        with torch.no_grad():
            audio_all = model.model.embed_audio(
                inputs_embeds=input_features.to(torch.bfloat16)
            ).float().numpy()
        audio_valid = audio_all[input_features_mask.numpy()]
        print(f"  audio_embeds shape: {audio_valid.shape}, 耗时: {(time.time()-t1)*1000:.1f} ms")
        merged_embeds[0, audio_mask, :] = audio_valid

    print(f"合并后 merged_embeds shape: {merged_embeds.shape}")

    # Step 4: LM autoregressive generation (greedy, with bidirectional mask for vision)
    max_new_tokens = 8
    print(f"Step 4: LM 自回归生成 (max_new_tokens={max_new_tokens})...")

    seq_len = merged_embeds.shape[1]
    generated_token_ids = []

    # Build bidirectional attention mask for vision tokens (same as model.model.forward)
    from transformers.models.gemma4_unified.modeling_gemma4_unified import get_block_sequence_ids_for_mask
    from transformers.masking_utils import create_masks_for_generate
    block_sequence_ids = get_block_sequence_ids_for_mask(mm_token_type_ids, device="cpu")

    # Prefill
    t1 = time.time()
    with torch.no_grad():
        merged_tensor = torch.from_numpy(merged_embeds).to(torch.bfloat16)
        attn_mask = torch.ones(1, seq_len, dtype=torch.long)
        pos_ids = torch.arange(seq_len).unsqueeze(0)

        causal_mask_mapping = create_masks_for_generate(
            config=model.config.get_text_config(),
            inputs_embeds=merged_tensor,
            attention_mask=attn_mask,
            past_key_values=None,
            position_ids=pos_ids,
            block_sequence_ids=block_sequence_ids,
        )

        lm_out = model.model.language_model(
            inputs_embeds=merged_tensor, attention_mask=causal_mask_mapping,
            position_ids=pos_ids, use_cache=True,
        )
        logits = model.lm_head(lm_out.last_hidden_state).float()
        next_token_id = int(torch.argmax(logits[0, -1, :]).item())
        generated_token_ids.append(next_token_id)
        past = lm_out.past_key_values
    t2 = time.time()
    print(f"  prefill: token_id={next_token_id}, 耗时: {(t2-t1)*1000:.0f} ms")

    # Decode with KV cache
    current_pos = seq_len
    for step in range(1, max_new_tokens):
        if next_token_id == 1:
            break
        t1 = time.time()
        with torch.no_grad():
            token_embed = model.model.language_model.embed_tokens(
                torch.tensor([[next_token_id]])
            )
            attn_mask = torch.ones(1, current_pos + 1, dtype=torch.long)
            pos_ids_step = torch.tensor([[current_pos]])

            causal_mask_mapping = create_masks_for_generate(
                config=model.config.get_text_config(),
                inputs_embeds=token_embed,
                attention_mask=attn_mask,
                past_key_values=past,
                position_ids=pos_ids_step,
            )

            lm_out = model.model.language_model(
                inputs_embeds=token_embed, attention_mask=causal_mask_mapping,
                position_ids=pos_ids_step, past_key_values=past, use_cache=True,
            )
            logits = model.lm_head(lm_out.last_hidden_state).float()
            next_token_id = int(torch.argmax(logits[0, -1, :]).item())
            generated_token_ids.append(next_token_id)
            past = lm_out.past_key_values
            current_pos += 1
        t2 = time.time()
        print(f"  step {step}: token_id={next_token_id}, 耗时: {(t2-t1)*1000:.0f} ms")

    response = processor.decode(generated_token_ids, skip_special_tokens=True)
    print(f"\n模型回复 (Torch Pipeline): {response}")
    return response


def run_openvino():
    """OpenVINO 推理路径：使用导出的 OV 模型逐步执行"""
    import time
    import openvino as ov
    import openvino_tokenizers  # registers tokenizer ops extension

    OV_EXPORT_DIR = "./ov_exported_gemma4"

    print("\n" + "=" * 60)
    print("== [OpenVINO] 推理")
    print("=" * 60)

    core = ov.Core()

    # --- Load OV models ---
    print("加载 OV 模型...")
    t0 = time.time()
    compiled_vision = core.compile_model(
        f"{OV_EXPORT_DIR}/openvino_vision_embeds_model_int8.xml", "CPU"
    )
    compiled_lm = core.compile_model(
        f"{OV_EXPORT_DIR}/openvino_language_model_int4.xml", "CPU"
    )
    compiled_text_emb = core.compile_model(
        f"{OV_EXPORT_DIR}/openvino_text_embeddings_model.xml", "CPU"
    )
    compiled_detokenizer = core.compile_model(
        f"{OV_EXPORT_DIR}/openvino_detokenizer.xml", "CPU"
    )
    print(f"  模型加载耗时: {(time.time()-t0)*1000:.0f} ms")

    # --- Step 1: 获取 processor 的输出 ---
    input_ids = inputs["input_ids"]  # [1, seq_len]
    attention_mask = inputs["attention_mask"]  # [1, seq_len]
    mm_token_type_ids = inputs["mm_token_type_ids"]  # [1, seq_len]
    mm_ids_np = mm_token_type_ids.numpy()[0]

    # --- Step 2: 计算 text embeddings (OV) ---
    print("计算 text embeddings (OV)...")
    t1 = time.time()
    ov_text_emb_result = compiled_text_emb({"input_ids": input_ids.numpy()})
    text_embeds = ov_text_emb_result[compiled_text_emb.output(0)]  # [1, seq_len, 3840]
    print(f"  text_embeds shape: {text_embeds.shape}, 耗时: {(time.time()-t1)*1000:.1f} ms")

    merged_embeds = text_embeds.copy()

    # --- Step 3: 计算 vision embeddings (OV) ---
    image_mask = (mm_ids_np == 1)
    if image_mask.any() and "pixel_values" in inputs:
        print("计算 vision embeddings (OV)...")
        pixel_values = inputs["pixel_values"]
        image_position_ids = inputs["image_position_ids"]
        t1 = time.time()
        ov_vision_result = compiled_vision({
            "pixel_values": pixel_values.numpy().astype(np.float32),
            "image_position_ids": image_position_ids.numpy(),
        })
        vision_embeds_all = ov_vision_result[compiled_vision.output(0)]
        # Strip padding patches (where image_position_ids == -1 on both axes)
        padding_mask = (image_position_ids.numpy() == -1).all(axis=-1)
        vision_embeds_valid = vision_embeds_all[~padding_mask]
        print(f"  vision_embeds shape: {vision_embeds_valid.shape}, 耗时: {(time.time()-t1)*1000:.1f} ms")
        merged_embeds[0, image_mask, :] = vision_embeds_valid

    # --- Step 3b: 计算 audio embeddings (OV) ---
    audio_mask = (mm_ids_np == 3)
    if audio_mask.any() and "input_features" in inputs:
        print("计算 audio embeddings (OV)...")
        compiled_audio = core.compile_model(
            f"{OV_EXPORT_DIR}/openvino_audio_embeds_model.xml", "CPU"
        )
        input_features = inputs["input_features"].numpy().astype(np.float32)
        input_features_mask = inputs["input_features_mask"].numpy()

        t1 = time.time()
        ov_audio_result = compiled_audio({"audio_features": input_features})
        audio_embeds_all = ov_audio_result[compiled_audio.output(0)]
        audio_embeds_valid = audio_embeds_all[input_features_mask]
        print(f"  audio_embeds shape: {audio_embeds_valid.shape}, 耗时: {(time.time()-t1)*1000:.1f} ms")
        merged_embeds[0, audio_mask, :] = audio_embeds_valid

    # --- Step 4: Merge complete ---
    print(f"合并后 merged_embeds shape: {merged_embeds.shape}")

    # --- Step 5: 自回归生成 (stateful LM with 4D attention mask, greedy) ---
    max_new_tokens = 20
    print(f"开始自回归生成 (stateful, max_new_tokens={max_new_tokens})...")

    seq_len = merged_embeds.shape[1]

    # Build 4D bidirectional attention mask for prefill:
    # - Causal (lower triangular) for text tokens
    # - Bidirectional within vision/audio token blocks
    # mm_ids_np: 0=text, 1=image, 2=video, 3=audio
    prefill_mask_4d = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))  # causal base
    # Make vision tokens (mm_ids==1) bidirectional within their block
    vision_positions = np.where(mm_ids_np == 1)[0]
    if len(vision_positions) > 0:
        # Find contiguous blocks of vision tokens
        breaks = np.where(np.diff(vision_positions) != 1)[0] + 1
        blocks = np.split(vision_positions, breaks)
        for block in blocks:
            # Within this block, all tokens can attend to each other
            for i in block:
                for j in block:
                    prefill_mask_4d[i, j] = 1.0
    prefill_mask_4d = prefill_mask_4d.reshape(1, 1, seq_len, seq_len)
    print(f"  Prefill 4D mask shape: {prefill_mask_4d.shape}")

    # Initialize stateful LM
    lm_infer_request = compiled_lm.create_infer_request()
    lm_infer_request.reset_state()

    generated_token_ids = []

    # Prefill: feed entire merged embeddings with bidirectional mask
    t1 = time.time()
    lm_infer_request.infer({
        "inputs_embeds": merged_embeds.astype(np.float32),
        "attention_mask": prefill_mask_4d,
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
    })
    logits = lm_infer_request.get_tensor("logits").data
    next_token_id = int(np.argmax(logits[0, -1, :]))
    generated_token_ids.append(next_token_id)
    t2 = time.time()
    print(f"  prefill: token_id={next_token_id}, 耗时: {(t2-t1)*1000:.0f} ms")

    # Decode: generate remaining tokens one by one
    # Decode mask: [1, 1, 1, past+1] — new token can attend to all past tokens
    current_pos = seq_len
    for step in range(1, max_new_tokens):
        if next_token_id == 1:  # EOS
            break

        t1 = time.time()
        next_ids = np.array([[next_token_id]], dtype=np.int64)
        new_token_embed = compiled_text_emb({"input_ids": next_ids})[compiled_text_emb.output(0)]

        # Decode mask: new token attends to all past (causal, no bidirectional needed)
        decode_mask_4d = np.ones((1, 1, 1, current_pos + 1), dtype=np.float32)

        lm_infer_request.infer({
            "inputs_embeds": new_token_embed.astype(np.float32),
            "attention_mask": decode_mask_4d,
            "position_ids": np.array([[current_pos]], dtype=np.int64),
        })
        logits = lm_infer_request.get_tensor("logits").data
        next_token_id = int(np.argmax(logits[0, -1, :]))
        generated_token_ids.append(next_token_id)
        current_pos += 1
        t2 = time.time()
        print(f"  step {step}: token_id={next_token_id}, 耗时: {(t2-t1)*1000:.0f} ms")

    # --- Step 6: Detokenize ---
    print("解码生成的 token...")
    token_ids_np = np.array([generated_token_ids], dtype=np.int64)
    detok_result = compiled_detokenizer(token_ids_np)
    decoded_text = list(detok_result.values())[0][0]
    print(f"\n模型回复 (OV): {decoded_text}")

    decoded_by_processor = processor.decode(generated_token_ids, skip_special_tokens=True)
    print(f"模型回复 (processor decode): {decoded_by_processor}")

    return decoded_by_processor


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch", action="store_true", help="Run torch model.generate()")
    parser.add_argument("--torch-pipeline", action="store_true", help="Run torch step-by-step pipeline (for comparison)")
    parser.add_argument("--ov", action="store_true", help="Run OpenVINO inference")
    args = parser.parse_args()

    if not args.torch and not args.torch_pipeline and not args.ov:
        args.torch = True
        args.ov = True

    if args.torch:
        torch_result = run_torch()

    if args.torch_pipeline:
        torch_pipeline_result = run_torch_pipeline()

    if args.ov:
        ov_result = run_openvino()

    # Print comparison if multiple modes ran
    results = {}
    if args.torch:
        results["Torch generate()"] = torch_result
    if args.torch_pipeline:
        results["Torch pipeline"] = torch_pipeline_result
    if args.ov:
        results["OpenVINO"] = ov_result
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("== 结果对比")
        print("=" * 60)
        for name, result in results.items():
            print(f"  {name:20s}: {result}")
