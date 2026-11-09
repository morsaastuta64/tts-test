import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
)

ref_output = ""
with open('output.txt', 'r') as file:
    ref_output = file.read().replace('\n', ' ')

wavs, sr = tts.generate_custom_voice(
    text=ref_output,
    speaker="speaker",
)
sf.write("output.wav", wavs[0], sr)