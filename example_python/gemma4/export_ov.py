"""
Gemma4 embed_vision / embed_audio / tokenizer / language_model => OpenVINO conversion & inference.

Usage:
  python example_ov.py --export      # export tokenizer + embed_vision + embed_audio to OV
  python example_ov.py --export-lm   # export language_model (input_embeds -> logits) to OV
  python example_ov.py --infer       # run OV inference and compare with torch reference
  python example_ov.py --infer-lm    # run OV LM inference and compare with torch reference
"""

import os
import argparse
import copy
import time

import torch
import numpy as np
from PIL import Image

import torch.nn as nn

import openvino as ov
from openvino_tokenizers import convert_tokenizer
from nncf import compress_weights, CompressWeightsMode

MODEL_ID = "/home/xiping/mygithub/profiling_ov_genai/models/models/google/gemma-4-12B-it"
MODEL_ID = "/home/xiping/mygithub/profiling_ov_genai/models/models/google/gemma-4-12B-it"
OV_EXPORT_DIR = "./ov_exported_gemma4"
IMAGE_PATH = "./cat_120_100.png"


def export_to_ov():
    """Export processor.tokenizer, model.model.embed_vision, model.model.embed_audio to OV."""
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    print("== Loading processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    )
    model.eval()

    os.makedirs(OV_EXPORT_DIR, exist_ok=True)

    # --- 0. Copy config files from original model ---
    import shutil
    import glob
    print("== Copying config files from original model...")
    for pattern in ["*.json", "*.jinja"]:
        for src_file in glob.glob(os.path.join(MODEL_ID, pattern)):
            dst_file = os.path.join(OV_EXPORT_DIR, os.path.basename(src_file))
            shutil.copy2(src_file, dst_file)
            print(f"  Copied {os.path.basename(src_file)}")

    # --- 1. Export tokenizer ---
    print("== Converting tokenizer to OV...")
    ov_tokenizer_model, ov_detokenizer_model = convert_tokenizer(
        processor.tokenizer, with_detokenizer=True
    )
    ov.save_model(ov_tokenizer_model, os.path.join(OV_EXPORT_DIR, "openvino_tokenizer.xml"))
    ov.save_model(ov_detokenizer_model, os.path.join(OV_EXPORT_DIR, "openvino_detokenizer.xml"))
    print(f"  Saved tokenizer => {OV_EXPORT_DIR}/openvino_tokenizer.xml")
    print(f"  Saved detokenizer => {OV_EXPORT_DIR}/openvino_detokenizer.xml")

    # --- 2. Export text embeddings (embed_tokens) ---
    print("== Converting embed_tokens to OV...")
    embed_tokens_fp32 = copy.deepcopy(model.model.language_model.embed_tokens).float()
    embed_tokens_fp32.eval()

    dummy_input_ids = torch.tensor([[2, 9259, 1902]], dtype=torch.long)
    ov_text_emb_model = ov.convert_model(
        embed_tokens_fp32,
        example_input=dummy_input_ids,
    )
    ov_text_emb_model.inputs[0].set_names({"input_ids"})
    ov_text_emb_model.outputs[0].set_names({"text_embeddings"})

    ov.save_model(ov_text_emb_model, os.path.join(OV_EXPORT_DIR, "openvino_text_embeddings_model.xml"),
                  compress_to_fp16=True)
    print(f"  Saved => {OV_EXPORT_DIR}/openvino_text_embeddings_model.xml (FP16)")

    # --- 3. Export embed_vision ---
    print("== Converting embed_vision to OV...")
    image = Image.open(IMAGE_PATH).convert("RGB")
    prompt = "<|im_start|>user\n<|image|>\nWhat is shown in this image?<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Use float32 for tracing to avoid bf16 issues in OV
    pixel_values = inputs["pixel_values"].to(torch.float32)  # [1, 280, 6912]
    image_position_ids = inputs["image_position_ids"]  # [1, 280, 2]

    # Convert model to float32 for export (weights will be compressed later)
    embed_vision_fp32 = copy.deepcopy(model.model.embed_vision).float()
    embed_vision_fp32.eval()

    ov_vision_model = ov.convert_model(
        embed_vision_fp32,
        example_input=(pixel_values, image_position_ids),
    )
    # Set friendly names
    ov_vision_model.inputs[0].set_names({"pixel_values"})
    ov_vision_model.inputs[1].set_names({"image_position_ids"})
    ov_vision_model.outputs[0].set_names({"vision_embeddings"})

    # Save FP16
    ov.save_model(ov_vision_model, os.path.join(OV_EXPORT_DIR, "openvino_vision_embeds_model.xml"),
                  compress_to_fp16=True)
    print(f"  Saved => {OV_EXPORT_DIR}/openvino_vision_embeds_model.xml (FP16)")

    # Save INT8
    ov_vision_model_int8 = compress_weights(
        copy.deepcopy(ov_vision_model), mode=CompressWeightsMode.INT8_SYM
    )
    ov.save_model(ov_vision_model_int8, os.path.join(OV_EXPORT_DIR, "openvino_vision_embeds_model_int8.xml"))
    print(f"  Saved => {OV_EXPORT_DIR}/openvino_vision_embeds_model_int8.xml (INT8)")

    # --- 4. Export embed_audio ---
    print("== Converting embed_audio to OV...")
    dummy_audio_input = torch.randn(1, 100, 640, dtype=torch.float32)

    embed_audio_fp32 = copy.deepcopy(model.model.embed_audio).float()
    embed_audio_fp32.eval()

    ov_audio_model = ov.convert_model(
        embed_audio_fp32,
        example_input=dummy_audio_input,
    )
    ov_audio_model.inputs[0].set_names({"audio_features"})
    ov_audio_model.outputs[0].set_names({"audio_embeddings"})

    # Save FP16
    ov.save_model(ov_audio_model, os.path.join(OV_EXPORT_DIR, "openvino_audio_embeds_model.xml"),
                  compress_to_fp16=True)
    print(f"  Saved => {OV_EXPORT_DIR}/openvino_audio_embeds_model.xml (FP16)")

    # Save INT8
    ov_audio_model_int8 = compress_weights(
        copy.deepcopy(ov_audio_model), mode=CompressWeightsMode.INT8_SYM
    )
    ov.save_model(ov_audio_model_int8, os.path.join(OV_EXPORT_DIR, "openvino_audio_embeds_model_int8.xml"))
    print(f"  Saved => {OV_EXPORT_DIR}/openvino_audio_embeds_model_int8.xml (INT8)")

    # --- 5. Save torch reference outputs for verification ---
    print("== Generating torch reference outputs...")
    with torch.no_grad():
        ref_vision_out = embed_vision_fp32(pixel_values, image_position_ids)
        ref_audio_out = embed_audio_fp32(dummy_audio_input)

    np.save(os.path.join(OV_EXPORT_DIR, "ref_vision_out.npy"), ref_vision_out.numpy())
    np.save(os.path.join(OV_EXPORT_DIR, "ref_audio_out.npy"), ref_audio_out.numpy())
    np.save(os.path.join(OV_EXPORT_DIR, "ref_pixel_values.npy"), pixel_values.numpy())
    np.save(os.path.join(OV_EXPORT_DIR, "ref_image_position_ids.npy"), image_position_ids.numpy())
    np.save(os.path.join(OV_EXPORT_DIR, "ref_audio_input.npy"), dummy_audio_input.numpy())
    print("  Saved reference numpy arrays for verification.")

    print("\n== Export complete!")


class LanguageModelWrapper(nn.Module):
    """Wraps language_model + lm_head: inputs_embeds -> logits (no KV cache, prefill mode)."""

    def __init__(self, language_model, lm_head):
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head

    def forward(self, inputs_embeds, attention_mask, position_ids):
        lm_out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = self.lm_head(lm_out.last_hidden_state)
        return logits


class LanguageModelWithPastWrapper(nn.Module):
    """Wraps language_model + lm_head with KV cache (for stateful export)."""

    def __init__(self, language_model, lm_head, config):
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head
        self.num_layers = config.num_hidden_layers

    def forward(self, inputs_embeds, attention_mask, position_ids, *past_kv_flat):
        from transformers.cache_utils import DynamicCache
        past = DynamicCache(config=self.language_model.config)
        for i in range(self.num_layers):
            past.update(key_states=past_kv_flat[i * 2], value_states=past_kv_flat[i * 2 + 1], layer_idx=i)

        out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
        )
        logits = self.lm_head(out.last_hidden_state)

        new_past = out.past_key_values
        output_kv = []
        for i in range(self.num_layers):
            layer = new_past.layers[i]
            output_kv.append(layer.keys)
            output_kv.append(layer.values)
        return (logits, *output_kv)


def _patch_dynamic_cache():
    """Monkey-patch DynamicLayer.update to avoid 0-rank tensor issue during OV tracing."""
    from transformers.cache_utils import DynamicLayer

    def _patched_update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.dtype = key_states.dtype
            self.device = key_states.device
            self.keys = key_states
            self.values = value_states
            self.is_initialized = True
            return self.keys, self.values
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    DynamicLayer.update = _patched_update


def export_lm_to_ov():
    """Export model.model.language_model + lm_head to OV as stateful model (with KV cache).
    Input: inputs_embeds, attention_mask, position_ids (KV cache is internal state).
    Output: logits [batch, seq_len, vocab_size].
    """
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    from openvino import passes

    print("== Loading model for LM export...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    )
    model.eval()

    os.makedirs(OV_EXPORT_DIR, exist_ok=True)

    config = model.config.text_config
    num_layers = config.num_hidden_layers

    # Patch DynamicCache to work with OV tracing
    _patch_dynamic_cache()

    # Create wrapper with KV cache, convert to float32
    wrapper = LanguageModelWithPastWrapper(model.model.language_model, model.lm_head, config).float()
    wrapper.eval()

    # Trace with non-zero past (decode mode)
    past_len = 3
    seq_len = 1
    dummy_embeds = torch.randn(1, seq_len, 3840, dtype=torch.float32)
    dummy_attn_mask = torch.ones(1, past_len + seq_len, dtype=torch.long)
    dummy_pos_ids = torch.tensor([[past_len]], dtype=torch.long)

    past_kv_flat = []
    for i in range(num_layers):
        lt = config.layer_types[i]
        if lt == "full_attention":
            kv_heads, head_dim = 1, 512
        else:
            kv_heads, head_dim = 8, 256
        past_kv_flat.append(torch.randn(1, kv_heads, past_len, head_dim, dtype=torch.float32))
        past_kv_flat.append(torch.randn(1, kv_heads, past_len, head_dim, dtype=torch.float32))

    print("== Testing wrapper forward...")
    with torch.no_grad():
        outputs = wrapper(dummy_embeds, dummy_attn_mask, dummy_pos_ids, *past_kv_flat)
    print(f"  Logits shape: {outputs[0].shape}")

    print("== Converting language model to OV (this may take a while)...")
    ov_lm_model = ov.convert_model(
        wrapper,
        example_input=(dummy_embeds, dummy_attn_mask, dummy_pos_ids, *past_kv_flat),
    )

    # Name inputs/outputs
    ov_lm_model.inputs[0].set_names({"inputs_embeds"})
    ov_lm_model.inputs[1].set_names({"attention_mask"})
    ov_lm_model.inputs[2].set_names({"position_ids"})
    for i in range(num_layers):
        ov_lm_model.inputs[3 + i * 2].set_names({f"past_key.{i}"})
        ov_lm_model.inputs[3 + i * 2 + 1].set_names({f"past_value.{i}"})
    ov_lm_model.outputs[0].set_names({"logits"})
    for i in range(num_layers):
        ov_lm_model.outputs[1 + i * 2].set_names({f"present_key.{i}"})
        ov_lm_model.outputs[1 + i * 2 + 1].set_names({f"present_value.{i}"})

    # Fix KV input shapes: set batch=1, kv_heads=fixed, seq=dynamic, head_dim=fixed
    # This ensures state variables have proper shapes for benchmark_app
    print("== Setting KV input partial shapes...")
    for i in range(num_layers):
        lt = config.layer_types[i]
        if lt == "full_attention":
            kv_heads, head_dim = 1, 512
        else:
            kv_heads, head_dim = 8, 256
        kv_shape = ov.PartialShape([1, kv_heads, -1, head_dim])
        ov_lm_model.inputs[3 + i * 2].get_node().set_partial_shape(kv_shape)
        ov_lm_model.inputs[3 + i * 2 + 1].get_node().set_partial_shape(kv_shape)
    ov_lm_model.validate_nodes_and_infer_types()

    # Apply MakeStateful: KV inputs/outputs become internal state
    print("== Applying MakeStateful transform...")
    state_pairs = {}
    for i in range(num_layers):
        state_pairs[f"past_key.{i}"] = f"present_key.{i}"
        state_pairs[f"past_value.{i}"] = f"present_value.{i}"

    manager = passes.Manager()
    manager.register_pass(passes.MakeStateful(state_pairs))
    manager.run_passes(ov_lm_model)
    print(f"  Stateful model: {len(ov_lm_model.inputs)} inputs, {len(ov_lm_model.outputs)} outputs")

    # Save FP16
    print("== Saving stateful FP16 model...")
    ov.save_model(ov_lm_model, os.path.join(OV_EXPORT_DIR, "openvino_language_model.xml"),
                  compress_to_fp16=True)
    print(f"  Saved => {OV_EXPORT_DIR}/openvino_language_model.xml (FP16, stateful)")

    # Save INT4
    print("== Compressing to INT4_SYM...")
    ov_lm_model_int4 = compress_weights(
        copy.deepcopy(ov_lm_model), mode=CompressWeightsMode.INT4_SYM
    )
    ov.save_model(ov_lm_model_int4, os.path.join(OV_EXPORT_DIR, "openvino_language_model_int4.xml"))
    print(f"  Saved => {OV_EXPORT_DIR}/openvino_language_model_int4.xml (INT4, stateful)")

    print("\n== LM export complete!")


def infer_lm_ov(device="CPU"):
    """Load stateful OV language model and test autoregressive inference."""
    print(f"== Running OV stateful LM inference on device: {device}")
    core = ov.Core()

    model_path = os.path.join(OV_EXPORT_DIR, "openvino_language_model_int4.xml")
    if not os.path.exists(model_path):
        model_path = os.path.join(OV_EXPORT_DIR, "openvino_language_model.xml")
    print(f"== Loading: {model_path}")
    compiled_lm = core.compile_model(model_path, device)
    infer_request = compiled_lm.create_infer_request()

    # Generate a few tokens from random embeddings to test stateful behavior
    seq_len = 5
    dummy_embeds = np.random.randn(1, seq_len, 3840).astype(np.float32)

    # Reset KV cache states (shapes are already correct from export)
    infer_request.reset_state()

    # Prefill
    print(f"== Prefill ({seq_len} tokens)...")
    t1 = time.time()
    infer_request.infer({
        "inputs_embeds": dummy_embeds,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
    })
    logits = infer_request.get_tensor("logits").data
    t2 = time.time()
    first_token = int(np.argmax(logits[0, -1, :]))
    print(f"  Prefill time: {(t2-t1)*1000:.0f} ms, first token: {first_token}")
    print(f"  Logits shape: {logits.shape}")

    # Decode steps
    print("== Decode (5 tokens)...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # For decode, we need embed_tokens to get the new token's embedding
    compiled_text_emb = core.compile_model(
        os.path.join(OV_EXPORT_DIR, "openvino_text_embeddings_model.xml"), "CPU"
    )

    generated = [first_token]
    current_pos = seq_len
    for step in range(5):
        t1 = time.time()
        # Get embedding for the last generated token
        token_ids = np.array([[generated[-1]]], dtype=np.int64)
        token_emb = compiled_text_emb({"input_ids": token_ids})[compiled_text_emb.output(0)]

        infer_request.infer({
            "inputs_embeds": token_emb,
            "attention_mask": np.ones((1, current_pos + 1), dtype=np.int64),
            "position_ids": np.array([[current_pos]], dtype=np.int64),
        })
        logits = infer_request.get_tensor("logits").data
        next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)
        current_pos += 1
        t2 = time.time()
        print(f"    step {step}: token={next_token}, time={((t2-t1)*1000):.0f} ms")

        if next_token == 1:  # EOS
            break

    decoded = processor.decode(generated, skip_special_tokens=True)
    print(f"\n  Generated tokens: {generated}")
    print(f"  Decoded: {decoded}")

    print("\n== OV stateful LM inference complete!")


def infer_ov(device="CPU"):
    """Load OV models and run inference, comparing with torch reference."""
    from transformers import AutoProcessor

    print(f"== Running OV inference on device: {device}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    core = ov.Core()

    # --- 1. Tokenizer ---
    print("== Loading OV tokenizer...")
    compiled_tokenizer = core.compile_model(
        os.path.join(OV_EXPORT_DIR, "openvino_tokenizer.xml"), "CPU"
    )
    test_text = "Hello, this is a test."
    ov_tok_result = compiled_tokenizer(ov.Tensor(np.array([test_text]).reshape(1)))
    ov_input_ids = ov_tok_result["input_ids"]
    ref_input_ids = processor.tokenizer(test_text, return_tensors="np")["input_ids"]
    print(f"  OV tokenizer input_ids:  {ov_input_ids[0][:10]}")
    print(f"  Ref tokenizer input_ids: {ref_input_ids[0][:10]}")
    tok_match = np.array_equal(
        ov_input_ids[0][: ref_input_ids.shape[1]], ref_input_ids[0]
    )
    print(f"  Tokenizer match: {tok_match}")

    # --- 1b. Detokenizer ---
    print("\n== Loading OV detokenizer...")
    compiled_detokenizer = core.compile_model(
        os.path.join(OV_EXPORT_DIR, "openvino_detokenizer.xml"), "CPU"
    )
    detok_result = compiled_detokenizer(ov_input_ids)
    detok_output = list(detok_result.values())[0]
    print(f"  Detokenized: {detok_output}")

    # --- 2. embed_vision ---
    print("\n== Loading OV embed_vision (FP16)...")
    compiled_vision = core.compile_model(
        os.path.join(OV_EXPORT_DIR, "openvino_vision_embeds_model.xml"), device
    )

    ref_pixel_values = np.load(os.path.join(OV_EXPORT_DIR, "ref_pixel_values.npy"))
    ref_image_position_ids = np.load(os.path.join(OV_EXPORT_DIR, "ref_image_position_ids.npy"))
    ref_vision_out = np.load(os.path.join(OV_EXPORT_DIR, "ref_vision_out.npy"))

    t1 = time.time()
    ov_vision_result = compiled_vision({
        "pixel_values": ref_pixel_values,
        "image_position_ids": ref_image_position_ids,
    })
    t2 = time.time()

    ov_vision_out = ov_vision_result[compiled_vision.output(0)]
    vision_max_diff = np.max(np.abs(ov_vision_out - ref_vision_out))
    vision_cos_sim = np.dot(ov_vision_out.flatten(), ref_vision_out.flatten()) / (
        np.linalg.norm(ov_vision_out.flatten()) * np.linalg.norm(ref_vision_out.flatten())
    )
    print(f"  embed_vision inference time: {(t2-t1)*1000:.2f} ms")
    print(f"  Max absolute diff vs torch ref: {vision_max_diff:.6f}")
    print(f"  Cosine similarity vs torch ref: {vision_cos_sim:.8f}")
    print(f"  Output shape: {ov_vision_out.shape}")

    # Benchmark
    print("  Benchmarking (10 runs)...")
    for i in range(10):
        t1 = time.time()
        compiled_vision({
            "pixel_values": ref_pixel_values,
            "image_position_ids": ref_image_position_ids,
        })
        t2 = time.time()
        print(f"    run {i}: {(t2-t1)*1000:.2f} ms")

    # --- 3. embed_audio ---
    print("\n== Loading OV embed_audio (FP16)...")
    compiled_audio = core.compile_model(
        os.path.join(OV_EXPORT_DIR, "openvino_audio_embeds_model.xml"), device
    )

    ref_audio_input = np.load(os.path.join(OV_EXPORT_DIR, "ref_audio_input.npy"))
    ref_audio_out = np.load(os.path.join(OV_EXPORT_DIR, "ref_audio_out.npy"))

    t1 = time.time()
    ov_audio_result = compiled_audio({"audio_features": ref_audio_input})
    t2 = time.time()

    ov_audio_out = ov_audio_result[compiled_audio.output(0)]
    audio_max_diff = np.max(np.abs(ov_audio_out - ref_audio_out))
    audio_cos_sim = np.dot(ov_audio_out.flatten(), ref_audio_out.flatten()) / (
        np.linalg.norm(ov_audio_out.flatten()) * np.linalg.norm(ref_audio_out.flatten())
    )
    print(f"  embed_audio inference time: {(t2-t1)*1000:.2f} ms")
    print(f"  Max absolute diff vs torch ref: {audio_max_diff:.6f}")
    print(f"  Cosine similarity vs torch ref: {audio_cos_sim:.8f}")
    print(f"  Output shape: {ov_audio_out.shape}")

    # Benchmark
    print("  Benchmarking (10 runs)...")
    for i in range(10):
        t1 = time.time()
        compiled_audio({"audio_features": ref_audio_input})
        t2 = time.time()
        print(f"    run {i}: {(t2-t1)*1000:.2f} ms")

    print("\n== OV inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma4 embed modules OV export & inference")
    parser.add_argument("--export", action="store_true", help="Export tokenizer + embeds to OV")
    parser.add_argument("--export-lm", action="store_true", help="Export language model to OV")
    parser.add_argument("--infer", action="store_true", help="Run OV embeds inference and compare")
    parser.add_argument("--infer-lm", action="store_true", help="Run OV LM inference and compare")
    parser.add_argument("--device", default="CPU", help="OV inference device (CPU/GPU)")
    args = parser.parse_args()

    if not args.export and not args.export_lm and not args.infer and not args.infer_lm:
        print("Please specify --export, --export-lm, --infer, or --infer-lm")
        parser.print_help()
    else:
        if args.export:
            export_to_ov()
        if args.export_lm:
            export_lm_to_ov()
        if args.infer:
            infer_ov(device=args.device)
        if args.infer_lm:
            infer_lm_ov(device=args.device)
