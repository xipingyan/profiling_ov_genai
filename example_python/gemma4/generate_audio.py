import os
# 如果没有安装 gTTS，请先在终端运行: pip install gTTS pydub
from gtts import gTTS
from pydub import AudioSegment

# 1. 将文本转为临时语音 (MP3格式)
text = "The speaker describes a fresh and peaceful day, enjoying a cup of coffee."
tts = gTTS(text=text, lang='en')
tts.save("temp.mp3")

# 2. 将音频转换为符合 Gemma 规范的 WAV 格式 (16kHz, 单声道)
audio = AudioSegment.from_mp3("temp.mp3")
audio = audio.set_frame_rate(16000).set_channels(1)

# 3. 导出文件
audio.export("journal1.wav", format="wav")

# 清理临时文件
os.remove("temp.mp3")
print("成功生成符合 Gemma 规范的 journal1.wav 测试音频！")