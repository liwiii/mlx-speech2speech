import ipdb
from transformers import (
        Qwen2TokenizerFast,
        WhisperFeatureExtractor
)
import librosa
from mlxs2s.text import TEXT, DEFAULT_CHAT_TEMPLATE

QWEN2_AUDIO_MODEL_PATH = '/Users/liwei15/huggingface_models/Qwen2-Audio-7B-Instruct/'

conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {"role": "user", "content": [
        {"type": "text", "text": "Stay alert and cautious, What does the person say?"},
        # {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"},
        {"type": "audio", "audio_url": "1272-128104-0000.flac"},
    ]},
]

# tokenizer = Qwen2TokenizerFast.from_pretrained(QWEN2_AUDIO_MODEL_PATH)
# text = tokenizer.apply_chat_template(
#                     conversation=conversation,
#                     chat_template=DEFAULT_CHAT_TEMPLATE,
#                     add_generation_prompt=True,
#                     tokenize=False)

# inputs = tokenizer(TEXT, padding=True, return_tensors="pt")
#
feature_extractor = WhisperFeatureExtractor.from_pretrained(QWEN2_AUDIO_MODEL_PATH)
audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                audios.append(
                    librosa.load(ele['audio_url'], sr=feature_extractor.sampling_rate)[0]
                )
# ipdb.set_trace()
audio_inputs = feature_extractor(audios,
                                 sampling_rate=feature_extractor.sampling_rate,
                                 return_attention_mask=True,
                                 padding="max_length",
                                 return_tensors="pt")

out = audio_inputs["input_features"].squeeze().to('cpu').numpy()
