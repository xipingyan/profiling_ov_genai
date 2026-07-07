import soundfile as sf
import importlib.util
import torch
from transformers import Qwen3OmniProcessor
from transformers import Qwen3OmniForConditionalGeneration
from qwen_omni_utils import process_mm_info

MODEL_PATH = "../qwen3_omni_model/"


def pick_attn_implementation():
    has_cuda = torch.cuda.is_available()
    has_flash_attn = importlib.util.find_spec("flash_attn") is not None

    if has_cuda and has_flash_attn:
        return "flash_attention_2"
    if has_cuda:
        return "sdpa"
    return "eager"


attn_impl = pick_attn_implementation()
print(f"Using attn_implementation = {attn_impl}")

model = Qwen3OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    attn_implementation=attn_impl,
)
processor = Qwen3OmniProcessor.from_pretrained(MODEL_PATH)

conversation = [
    {
        "role": "user",
        "content": [
            # {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
            # {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"},
            {"type": "image", "image": "./cars.jpg"},
            {"type": "audio", "audio": "./cough.wav"},
            {"type": "text", "text": "What can you see and hear? Answer in one short sentence."}
        ],
    },
]


# Set whether to use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, 
                   audio=audios, 
                   images=images, 
                   videos=videos, 
                   return_tensors="pt", 
                   padding=True, 
                   use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs, 
                                 speaker="f245", 
                                 thinker_return_dict_in_generate=True,
                                 use_audio_in_video=USE_AUDIO_IN_VIDEO,
                                 talker_max_new_tokens = 128)

text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)
print("\n===============================")
print("output text:", text)
if audio is not None:
    print("output audio saved to output.wav")
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )