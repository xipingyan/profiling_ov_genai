# 多模态模型转 OpenVINO 及推理 (以 Gemma4-12B-it 为例)

## 概述

将多模态 LLM 的各子模块分别转换为 OpenVINO 格式，其中 LLM 主体导出为 **stateful 模型**（KV cache 作为内部状态），然后用纯 OV 模型做端到端推理。

本例使用 Gemma4-12B-it，模型结构：
- `processor.tokenizer` — 文本 tokenizer
- `model.model.language_model.embed_tokens` — 文本 token embedding (含 scale)
- `model.model.embed_vision` — 图像 patch → embedding
- `model.model.embed_audio` — 音频特征 → embedding
- `model.model.language_model` + `model.lm_head` — LLM 主体 (48 层 transformer + logits head)

---

## 一、转模型核心技术

### 1. 各子模块转换方式

| 子模块 | 输入 | 输出 | 转换方法 |
|--------|------|------|----------|
| tokenizer | string | input_ids, attention_mask | `openvino_tokenizers.convert_tokenizer()` |
| embed_tokens | input_ids [B, seq] | embeddings [B, seq, hidden] | `ov.convert_model()` float32 tracing |
| embed_vision | pixel_values, image_position_ids | vision_embeds [B, patches, hidden] | `ov.convert_model()` float32 tracing |
| embed_audio | audio_features [B, N, 640] | audio_embeds [B, N, hidden] | `ov.convert_model()` float32 tracing |
| language_model + lm_head | inputs_embeds, attention_mask, position_ids | logits [B, seq, vocab] | **stateful export** (见下文) |

### 2. LLM 转 stateful 模型（关键步骤）

LLM 需要 KV cache 支持自回归生成。OV stateful 模型将 KV cache 作为内部 state，外部只传入 3 个 input，无需手动管理 96 个 KV tensor。

#### 步骤 2.1: 创建带 past_key_values 的 Wrapper

```python
class LanguageModelWithPastWrapper(nn.Module):
    """language_model + lm_head, with explicit KV cache inputs/outputs."""
    def __init__(self, language_model, lm_head, config):
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head
        self.num_layers = config.num_hidden_layers

    def forward(self, inputs_embeds, attention_mask, position_ids, *past_kv_flat):
        from transformers.cache_utils import DynamicCache
        past = DynamicCache(config=self.language_model.config)
        for i in range(self.num_layers):
            past.update(key_states=past_kv_flat[i*2], value_states=past_kv_flat[i*2+1], layer_idx=i)
        out = self.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past, use_cache=True,
        )
        logits = self.lm_head(out.last_hidden_state)
        # Flatten output KV
        new_past = out.past_key_values
        output_kv = []
        for i in range(self.num_layers):
            layer = new_past.layers[i]
            output_kv.append(layer.keys)
            output_kv.append(layer.values)
        return (logits, *output_kv)
```

#### 步骤 2.2: Monkey-patch DynamicCache（解决 OV tracing 0-rank tensor 问题）

transformers 的 `DynamicLayer.update()` 内部会创建空 tensor `torch.tensor([])` 再 cat，OV 的 torch frontend 无法处理这个 0-rank → 4-rank 的 concat。需要 patch：

```python
def _patch_dynamic_cache():
    from transformers.cache_utils import DynamicLayer
    def _patched_update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.dtype = key_states.dtype
            self.device = key_states.device
            self.keys = key_states      # 直接赋值，跳过 lazy_initialization
            self.values = value_states
            self.is_initialized = True
            return self.keys, self.values
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values
    DynamicLayer.update = _patched_update
```

#### 步骤 2.3: 用非零 past 做 tracing

必须用 **non-zero past_len** 做 tracing（如 past_len=3），不能用 0-length，否则 OV 的 concat 验证会失败：

```python
past_len = 3
seq_len = 1  # decode mode: 1 new token
dummy_embeds = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
dummy_attn_mask = torch.ones(1, past_len + seq_len, dtype=torch.long)
dummy_pos_ids = torch.tensor([[past_len]], dtype=torch.long)

# 构建 past KV（需要区分不同 attention type 的 head 数和 dim）
past_kv_flat = []
for i in range(num_layers):
    if layer_types[i] == "full_attention":
        kv_heads, head_dim = 1, 512      # Gemma4 的 global attention
    else:
        kv_heads, head_dim = 8, 256      # Gemma4 的 sliding attention
    past_kv_flat.append(torch.randn(1, kv_heads, past_len, head_dim))
    past_kv_flat.append(torch.randn(1, kv_heads, past_len, head_dim))

ov_model = ov.convert_model(wrapper, example_input=(dummy_embeds, dummy_attn_mask, dummy_pos_ids, *past_kv_flat))
```

#### 步骤 2.4: 命名 inputs/outputs 并应用 MakeStateful

```python
from openvino import passes

# 命名
ov_model.inputs[0].set_names({"inputs_embeds"})
ov_model.inputs[1].set_names({"attention_mask"})
ov_model.inputs[2].set_names({"position_ids"})
for i in range(num_layers):
    ov_model.inputs[3 + i*2].set_names({f"past_key.{i}"})
    ov_model.inputs[3 + i*2 + 1].set_names({f"past_value.{i}"})
ov_model.outputs[0].set_names({"logits"})
for i in range(num_layers):
    ov_model.outputs[1 + i*2].set_names({f"present_key.{i}"})
    ov_model.outputs[1 + i*2 + 1].set_names({f"present_value.{i}"})

# MakeStateful: 把 past_key/value input 和 present_key/value output 配对为内部 state
state_pairs = {}  # dict 形式: {input_name: output_name}
for i in range(num_layers):
    state_pairs[f"past_key.{i}"] = f"present_key.{i}"
    state_pairs[f"past_value.{i}"] = f"present_value.{i}"

manager = passes.Manager()
manager.register_pass(passes.MakeStateful(state_pairs))
manager.run_passes(ov_model)
# 结果: 模型只剩 3 inputs (inputs_embeds, attention_mask, position_ids) 和 1 output (logits)
```

#### 步骤 2.5: 固定 KV state 形状（使 benchmark_app 可用）

MakeStateful 后 state 默认是全动态 `{?,?,?,dim}`，benchmark_app 无法自动初始化。需要在 MakeStateful **之前**设置 KV input 的 partial shape：

```python
# 在 MakeStateful 之前执行
for i in range(num_layers):
    if layer_types[i] == "full_attention":
        kv_heads, head_dim = 1, 512
    else:
        kv_heads, head_dim = 8, 256
    kv_shape = ov.PartialShape([1, kv_heads, -1, head_dim])  # batch=1, heads固定, seq动态, dim固定
    ov_model.inputs[3 + i*2].get_node().set_partial_shape(kv_shape)      # past_key
    ov_model.inputs[3 + i*2 + 1].get_node().set_partial_shape(kv_shape)  # past_value
ov_model.validate_nodes_and_infer_types()
```

这样转出的 stateful 模型，state 初始形状为 `[1, kv_heads, 0, head_dim]`：
- `reset_state()` 后直接可用，无需手动设置 shape
- benchmark_app 可以直接测试

#### 步骤 2.6: 压缩并保存

```python
ov.save_model(ov_model, "openvino_language_model.xml", compress_to_fp16=True)

from nncf import compress_weights, CompressWeightsMode
ov_model_int4 = compress_weights(copy.deepcopy(ov_model), mode=CompressWeightsMode.INT4_SYM)
ov.save_model(ov_model_int4, "openvino_language_model_int4.xml")
```

### 4. 拷贝原始模型配置文件

导出 OV 模型的同时，需要把原始模型目录下的配置文件也拷贝到导出目录，供后续 processor/tokenizer 加载使用：

```python
import shutil
import glob

for pattern in ["*.json", "*.jinja"]:
    for src_file in glob.glob(os.path.join(MODEL_ID, pattern)):
        dst_file = os.path.join(OV_EXPORT_DIR, os.path.basename(src_file))
        shutil.copy2(src_file, dst_file)
```

需要拷贝的文件（以 Gemma4-12B-it 为例）：
- `config.json` — 模型架构配置（hidden_size, num_layers, layer_types 等）
- `generation_config.json` — 生成参数（max_length, eos_token_id 等）
- `tokenizer_config.json` — tokenizer 配置
- `tokenizer.json` — tokenizer vocab 和 merge rules
- `processor_config.json` — processor 配置（image/audio 处理参数）
- `chat_template.jinja` — chat template（apply_chat_template 需要）

这些文件使得导出目录可以直接用 `AutoProcessor.from_pretrained(OV_EXPORT_DIR)` 加载。

### 3. 推理 stateful 模型

使用 `InferRequest` 的 stateful API：

```python
compiled_lm = core.compile_model("openvino_language_model_int4.xml", "CPU")
lm_request = compiled_lm.create_infer_request()

# 重置 KV cache（导出时已固定 shape，reset_state() 即可）
lm_request.reset_state()

# Prefill: 一次性喂入整个 merged_embeds
lm_request.infer({
    "inputs_embeds": merged_embeds,          # [1, seq_len, 3840]
    "attention_mask": np.ones((1, seq_len)),  # [1, seq_len]
    "position_ids": np.arange(seq_len).reshape(1, -1),
})
logits = lm_request.get_tensor("logits").data
first_token = np.argmax(logits[0, -1, :])

# Decode: 逐 token 生成（KV cache 自动累积）
current_pos = seq_len
for step in range(max_new_tokens - 1):
    token_embed = compiled_text_emb({"input_ids": [[next_token]]})[0]  # [1, 1, 3840]
    lm_request.infer({
        "inputs_embeds": token_embed,
        "attention_mask": np.ones((1, current_pos + 1)),
        "position_ids": np.array([[current_pos]]),
    })
    logits = lm_request.get_tensor("logits").data
    next_token = np.argmax(logits[0, -1, :])
    current_pos += 1
```

---

## 二、通用注意事项

### 转换时的坑

1. **必须用 float32 tracing** — bf16 模型需要 `.float()` 后再转，否则 OV 会遇到 shape 问题
2. **DynamicCache 需要 patch** — transformers >= 5.x 的 DynamicCache 使用 `torch.tensor([])` 初始化，OV frontend 不支持 0-rank tensor 的 concat
3. **tracing 必须用非零 past** — 空 past (seq=0) 会导致 `Axis out of range` 错误
4. **MakeStateful 接受 dict** — `passes.MakeStateful({"input_name": "output_name"})` 不是 list of tuples
5. **KV shape 必须在 MakeStateful 前固定** — 用 `set_partial_shape([1, kv_heads, -1, head_dim])` 固定 batch 和 heads 维度，否则 state 初始形状为 `{0,0,0,dim}` 导致 benchmark_app 无法运行、推理时 concat 维度不匹配

### 精度差异来源

| 来源 | 影响程度 | 说明 |
|------|---------|------|
| bf16 → fp16 (LM) | 中 | 48 层累积误差，top-1 token 可能翻转 |
| INT4 量化 (LM) | 大 | logits 最大偏差 ~8，但语义通常正确 |
| INT8 量化 (vision) | 小 | cosine > 0.9999 |
| FP16 (embed_tokens) | 极小 | cosine > 0.99999 |

### 多模态 embedding 合并逻辑

```python
# mm_token_type_ids 只依赖 input_ids，标记每个 token 属于哪个模态:
# 0 = text, 1 = image (token_id=258880), 2 = video, 3 = audio (token_id=258881)

merged_embeds = text_embeds.copy()
if image_mask.any():
    # embed_vision 输出去掉 padding patches 后 scatter 到 image 位置
    merged_embeds[0, mm_ids == 1, :] = vision_embeds_valid
if audio_mask.any():
    # embed_audio 输出去掉 padding 后 scatter 到 audio 位置
    merged_embeds[0, mm_ids == 3, :] = audio_embeds_valid
```

---

## 三、端到端验证（Torch pipeline vs OV pipeline）

转模型后，**必须用 torch 走完全一样的 pipeline 做对比**，确认差异在可接受范围内。

### 验证命令

```bash
# 同时跑 torch pipeline 和 OV，自动对比结果
python example.py --torch-pipeline --ov

# 也可以和 model.generate() 对比
python example.py --torch --torch-pipeline --ov
```

### --torch-pipeline 做了什么

用原始 torch 模型，执行和 OV 完全一样的步骤：
1. `model.model.language_model.embed_tokens(input_ids)` → text_embeds
2. `model.model.embed_vision(pixel_values, image_position_ids)` → vision_embeds（去 padding）
3. `model.model.embed_audio(input_features)` → audio_embeds（去 padding）
4. 按 `mm_token_type_ids` 合并为 merged_embeds
5. `model.model.language_model(inputs_embeds=merged_embeds, use_cache=True)` → prefill
6. 逐 token decode（with KV cache）
7. `model.lm_head(hidden_states)` → logits → argmax

这确保和 OV 是**完全相同的数据流**，差异只来自精度（fp16/int4 vs bf16）。

### 预期对比结果

```
============================================================
== 结果对比
============================================================
  Torch generate()    : The image shows a gray and white tabby
  Torch pipeline      : The image shows a close-up of
  OpenVINO            : The image is very blurry and appears to
```

| 对比 | 差异原因 |
|------|---------|
| `Torch generate()` vs `Torch pipeline` | generate() 内部有 bidirectional attention mask 给 vision tokens，pipeline 简化版没加 → 差异正常 |
| `Torch pipeline` vs `OV (FP16 LM)` | bf16→fp16 精度损失，前 2-3 token 通常一致 |
| `Torch pipeline` vs `OV (INT4 LM)` | INT4 量化，前 1-2 token 一致，之后分叉 |

### 逐模块精度验证（已测结果）

| 模块 | Max Abs Diff | Cosine Similarity | 结论 |
|------|-------------|-------------------|------|
| embed_tokens (FP16) | 0.25 | 0.99999 | 极好 |
| embed_vision (INT8) | 0.50 | 0.99995 | 好 |
| embed_vision (FP16) | 0.25 | 0.99999 | 极好 |
| embed_audio (FP16) | 0.07 | 0.99999 | 极好 |
| LM logits (FP16, same input) | 0.875 | — | top-5 一致，top-1 可能翻转 |
| LM logits (INT4, same input) | 8.8 | — | top-5 基本一致，语义正确 |

---

## 四、命令参考

```bash
cd /home/xiping/mygithub/profiling_ov_genai/example_python/gemma4
source python-env/bin/activate

# 转模型
python export_ov.py --export      # tokenizer + embed_tokens + embed_vision + embed_audio + 配置文件
python export_ov.py --export-lm   # language_model (stateful, FP16 + INT4)

# 子模块精度验证
python export_ov.py --infer       # 验证 tokenizer + embeds 精度
python export_ov.py --infer-lm    # 验证 stateful LM (prefill + decode)

# 端到端推理与对比
python example.py --ov                     # 纯 OV 推理
python example.py --torch                  # Torch model.generate()
python example.py --torch-pipeline         # Torch 逐步 pipeline（与 OV 同流程）
python example.py --torch-pipeline --ov    # 对比: torch pipeline vs OV
```

---

## 五、性能参考 (CPU, Gemma4-12B-it INT4)

| 步骤 | 耗时 |
|------|------|
| 模型加载 | ~3200 ms |
| Text embedding | ~9 ms |
| Vision embedding | ~20 ms |
| Audio embedding | ~5 ms |
| LM prefill (~300 tokens) | ~2400 ms |
| LM decode (每步 1 token) | **~180 ms** |

对比无 KV cache 版本 decode ~900 ms/token，stateful 提速 **~5x**。

---

## 六、导出文件结构

```
ov_exported_gemma4/
├── config.json                          # 原始模型配置
├── generation_config.json               # 生成参数
├── tokenizer_config.json                # tokenizer 配置
├── tokenizer.json                       # tokenizer vocab
├── processor_config.json                # processor 配置
├── chat_template.jinja                  # chat template
├── openvino_tokenizer.xml / .bin        # OV tokenizer
├── openvino_detokenizer.xml / .bin      # OV detokenizer
├── openvino_text_embeddings_model.xml / .bin   # embed_tokens (FP16)
├── openvino_vision_embeds_model.xml / .bin     # embed_vision (FP16)
├── openvino_vision_embeds_model_int8.xml / .bin
├── openvino_audio_embeds_model.xml / .bin      # embed_audio (FP16)
├── openvino_audio_embeds_model_int8.xml / .bin
├── openvino_language_model.xml / .bin          # LM stateful (FP16, ~23GB)
├── openvino_language_model_int4.xml / .bin     # LM stateful (INT4, ~6.2GB)
└── ref_*.npy                                   # 参考数据
```

---

## 七、如何用这个 skill 转新模型

### 输入

用户提供：
1. **原始 torch 推理脚本** (`example.py`) — 包含模型加载、processor 调用、model.generate() 等
2. **需要转的子模块列表** — 如 "转 tokenizer, embed_vision, embed_audio, language_model"
3. **模型路径** (可选) — 如未提供，从脚本中的 MODEL_ID 获取

### 输出

生成两个文件：
1. **`export_ov.py`** — 转模型脚本（`--export` + `--export-lm`）
2. **`example.py`** (修改原文件) — 添加 `--ov` 和 `--torch-pipeline` 模式

### 工作流程（分步执行，逐步验证）

**重要：必须严格分步，每步完成后让用户验证，确认无误后再进入下一步。不要一次性做完所有事情。**

---

#### 第一步：分析模型结构，确定需要转的子模块

目标：列出所有需要转的子模块，以及每个子模块的输入输出 shape。

```python
# 常用探测命令
model = AutoModelForXxx.from_pretrained(MODEL_ID, ...)
for name, child in model.named_children():
    print(f"{name}: {type(child).__name__}")

config = model.config.text_config  # 或 model.config
print(f"num_layers={config.num_hidden_layers}")
print(f"num_kv_heads={config.num_key_value_heads}")
print(f"head_dim={config.head_dim}")
print(f"hidden_size={config.hidden_size}")
```

**交付物：** 列出需要转换的子模块清单表格，包括模块名、输入、输出、计划使用的 OV 文件名。

**⏸️ 等待用户确认后，进入第二步。**

---

#### 第二步：编写 `example.py --torch-pipeline`

目标：用原始 torch 模型，把推理拆成和未来 OV 一样的子步骤（embed_tokens → embed_vision → merge → LM with KV cache → decode）。**确保 `--torch-pipeline` 和原始 `--torch`（model.generate()）能得到基本一致的结果。**

这一步的意义：
- 验证我们对模型数据流的理解是正确的
- 建立 ground truth，后续每个 OV 模块都可以和这里的中间结果逐一对比
- 如果 `--torch-pipeline` 和 `--torch` 结果不一致，说明 pipeline 理解有误，必须先修正

```python
def run_torch_pipeline():
    model = AutoModelForXxx.from_pretrained(MODEL_ID, ...)
    # 1. embed_tokens → text_embeds
    # 2. embed_vision → vision_embeds (去 padding)
    # 3. embed_audio → audio_embeds (去 padding)
    # 4. merge (按 mm_token_type_ids scatter)
    # 5. language_model prefill (use_cache=True) → first token
    # 6. decode loop with past_key_values → remaining tokens
    # 7. lm_head → logits → argmax
```

**验证标准：**
```bash
python example.py --torch --torch-pipeline
# 两者结果应该基本一致（允许 bidirectional mask 等导致的轻微差异）
```

**⏸️ 等待用户确认 `--torch-pipeline` 结果正确后，进入第三步。**

---

#### 第三步：逐个转 OV 模型，每转一个验证一个

目标：编写 `export_ov.py`，按子模块逐个转换，**每个模块转完后立即验证精度**。

执行顺序：

**3a. 转 tokenizer + detokenizer**
```bash
python export_ov.py --export  # 只转 tokenizer 部分
# 验证：用 OV tokenizer encode 一段文字，和 processor.tokenizer 对比，必须完全一致
```

**3b. 转 embed_tokens**
```bash
# 验证：对同一组 input_ids，比较 torch embed_tokens 输出和 OV 输出
# 标准：cosine similarity > 0.9999
```

**3c. 转 embed_vision（如有）**
```bash
# 验证：对同一张图片的 pixel_values，比较 torch 和 OV 输出
# 标准：cosine similarity > 0.9999 (FP16), > 0.999 (INT8)
```

**3d. 转 embed_audio（如有）**
```bash
# 验证：对同一段音频的 input_features，比较 torch 和 OV 输出
# 标准：cosine similarity > 0.9999
```

**3e. 转 language_model（stateful）**
```bash
python export_ov.py --export-lm
# 验证：对同一个 merged_embeds 输入（从 torch pipeline 保存的 numpy），
# 比较 torch LM 和 OV LM 输出的 top-5 tokens
# 标准：top-5 集合应基本一致（顺序可以不同）
```

**⏸️ 每个子模块转完后，告知用户验证结果（cosine sim, max diff, top-k match）。全部通过后进入第四步。**

---

#### 第四步：编写 `example.py --ov` 端到端推理

目标：把所有 OV 模型串联起来，实现完整的端到端推理。

```python
def run_openvino():
    # 1. 加载所有 OV 模型
    # 2. 复用 processor 处理输入
    # 3. text_embeds = OV embed_tokens(input_ids)
    # 4. vision_embeds = OV embed_vision(pixel_values) (去 padding, scatter)
    # 5. audio_embeds = OV embed_audio(features) (去 padding, scatter)
    # 6. stateful LM: prefill + decode loop
    # 7. detokenize
```

**验证标准：**
```bash
python example.py --torch-pipeline --ov
# 对比结果:
# - 前 2-3 个 token 应该一致
# - 整体语义应该一致（都在描述同一件事）
# - 精度差异来自 FP16/INT4 量化，属于正常
```

**⏸️ 让用户确认最终 OV 结果是否可接受。**

---

### 关键判断点

| 需要判断的事项 | 如何获取信息 |
|---------------|-------------|
| 模型有几种 attention type | `config.layer_types` 或 `config.attn_implementation` |
| 每种 type 的 kv_heads 和 head_dim | `config.num_key_value_heads`, `config.head_dim`；如有 global attention 看 `num_global_key_value_heads`, `global_head_dim` |
| 是否有 embed_scale | 查看 `embed_tokens.forward` 源码 |
| vision padding 逻辑 | 查看 `get_image_features` 源码 |
| audio padding 逻辑 | 查看 `get_audio_features` 源码 |
| mm_token_type_ids 如何计算 | 查看 `processor.create_mm_token_type_ids` 源码 |
| lm_head 是否 tied | `config.tie_word_embeddings` |

### 代码模板

<details>
<summary>export_ov.py 模板</summary>

```python
def export_to_ov():
    # 0. 拷贝配置文件 (*.json, *.jinja)
    # 1. convert_tokenizer → openvino_tokenizer.xml, openvino_detokenizer.xml
    # 2. embed_tokens → openvino_text_embeddings_model.xml
    # 3. embed_vision → openvino_vision_embeds_model.xml (如有)
    # 4. embed_audio → openvino_audio_embeds_model.xml (如有)
    # 5. 其他 encoder → openvino_xxx_model.xml (如有)

def export_lm_to_ov():
    # 1. _patch_dynamic_cache()
    # 2. LanguageModelWithPastWrapper(language_model, lm_head, config)
    # 3. 构建 past_kv_flat (non-zero past_len)
    # 4. ov.convert_model(wrapper, example_input=...)
    # 5. 命名 inputs/outputs
    # 6. 设置 KV partial shape (固定 batch+heads)
    # 7. MakeStateful
    # 8. 保存 FP16 + INT4
```

</details>

<details>
<summary>run_openvino() 模板</summary>

```python
def run_openvino():
    core = ov.Core()

    # 1. 加载所有 OV 模型
    compiled_text_emb = core.compile_model(f"{OV_DIR}/openvino_text_embeddings_model.xml", "CPU")
    compiled_vision = core.compile_model(f"{OV_DIR}/openvino_vision_embeds_model_int8.xml", "CPU")
    compiled_audio = core.compile_model(f"{OV_DIR}/openvino_audio_embeds_model.xml", "CPU")
    compiled_lm = core.compile_model(f"{OV_DIR}/openvino_language_model_int4.xml", "CPU")
    compiled_detokenizer = core.compile_model(f"{OV_DIR}/openvino_detokenizer.xml", "CPU")

    # 2. 用原始 processor 处理输入（复用！）
    inputs = processor.apply_chat_template(messages, ...)

    # 3. Text embeddings
    text_embeds = compiled_text_emb({"input_ids": input_ids})[0]

    # 4. 多模态 embeddings (按 mm_token_type_ids 判断)
    merged_embeds = text_embeds.copy()
    if (mm_ids == 1).any() and "pixel_values" in inputs:
        vision_out = compiled_vision({...})[0]
        vision_valid = vision_out[~padding_mask]
        merged_embeds[0, mm_ids == 1, :] = vision_valid
    if (mm_ids == 3).any() and "input_features" in inputs:
        audio_out = compiled_audio({...})[0]
        audio_valid = audio_out[features_mask]
        merged_embeds[0, mm_ids == 3, :] = audio_valid

    # 5. Stateful LM: prefill + decode
    lm_request = compiled_lm.create_infer_request()
    lm_request.reset_state()

    # Prefill
    lm_request.infer({"inputs_embeds": merged_embeds, "attention_mask": ..., "position_ids": ...})
    next_token = np.argmax(lm_request.get_tensor("logits").data[0, -1, :])

    # Decode loop
    for step in range(max_new_tokens - 1):
        token_emb = compiled_text_emb({"input_ids": [[next_token]]})[0]
        lm_request.infer({"inputs_embeds": token_emb, "attention_mask": ..., "position_ids": [[pos]]})
        next_token = np.argmax(lm_request.get_tensor("logits").data[0, -1, :])

    # 6. Detokenize
    result = compiled_detokenizer(token_ids)[0]
```

</details>

---

## 八、适配新模型的 checklist

1. 确认模型的 attention 类型和 KV head 配置（`config.layer_types`, `num_key_value_heads`, `head_dim`）
2. 确认 embed_tokens 是否有 scale（Gemma 系列有 `* sqrt(hidden_size)`）
3. 确认多模态 token ID（`image_token_id`, `audio_token_id` 等）
4. 确认 vision embedder 的 padding 逻辑（Gemma4 用 `image_position_ids == -1`）
5. 确认 `lm_head` 是否与 `embed_tokens` weight tied（影响模型大小）
6. 如果模型有不同的 cache 实现（StaticCache, SlidingWindowCache），需要调整 patch 逻辑
7. 确认 processor 输出中哪些字段对应哪个模态（`pixel_values`, `input_features`, `mm_token_type_ids`）
8. 确认是否需要 bidirectional attention mask（影响 torch pipeline 和 generate() 的结果差异）

---

## 九、依赖

```
transformers>=5.10.0
openvino>=2025.0
openvino-tokenizers
nncf
safetensors
Pillow
numpy
torch
librosa          # (audio 处理需要)
```
