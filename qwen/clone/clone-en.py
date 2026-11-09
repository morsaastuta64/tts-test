import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

ref_audio = "input.wav"
ref_text  = ""
with open('input.txt', 'r') as file:
    ref_text = file.read().replace('\n', ' ')

ref_output = ""
with open('output.txt', 'r') as file:
    ref_output = file.read().replace('\n', ' ')

prompt_items = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=False,
)

arr_lang = ["English"] * 3
arr_outputs = []
for i in range(0, 3):
    arr_outputs.append(ref_output)

wavs, sr = model.generate_voice_clone(
    text=arr_outputs,
    language=arr_lang,
    voice_clone_prompt=prompt_items,
)

idx = 0
for wav in wavs:
    sf.write("output_" + str(idx) + ".wav", wav, sr)
    idx += 1