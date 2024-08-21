# from pathlib import Path
import ipdb
import librosa
from io import BytesIO
from urllib.request import urlopen
import torch
from transformers import (
        Qwen2AudioProcessor,
        Qwen2TokenizerFast,
        WhisperFeatureExtractor,
        Qwen2AudioForConditionalGeneration,
        BatchFeature,
        Qwen2AudioConfig,
        Qwen2AudioEncoderConfig,
        # Qwen2AudioMultiModalProjector,
        Qwen2AudioEncoder
)

DEFAULT_CHAT_TEMPLATE = (
            "{% set audio_count = namespace(value=0) %}"
            "{% for message in messages %}"
                "{% if loop.first and message['role'] != 'system' %}"
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "{% endif %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{% if message['content'] is string %}"
                    "{{ message['content'] }}<|im_end|>\n"
                "{% else %}"
                    "{% for content in message['content'] %}"
                        "{% if 'audio' in content or 'audio_url' in content %}"
                            "{% set audio_count.value = audio_count.value + 1 %}"
                            "Audio {{ audio_count.value }}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
                        "{% elif 'text' in content %}"
                            "{{ content['text'] }}"
                        "{% endif %}"
                    "{% endfor %}"
                    "<|im_end|>\n"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
            "{% endif %}"
)

QWEN2_AUDIO_MODEL_PATH = '/Users/liwei15/huggingface_models/Qwen2-Audio-7B-Instruct/'

conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {"role": "user", "content": [
        {"type": "text", "text": "Stay alert and cautious, What does the person say?"},
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"},
    ]},
]

# process = Qwen2AudioProcessor.from_pretrained(QWEN2_AUDIO_MODEL_PATH)
# ipdb.set_trace()
tokenizer = Qwen2TokenizerFast.from_pretrained(QWEN2_AUDIO_MODEL_PATH)
feature_extractor = WhisperFeatureExtractor.from_pretrained(QWEN2_AUDIO_MODEL_PATH)

# print(process.default_chat_template)
text = tokenizer.apply_chat_template(
                    conversation=conversation,
                    # chat_template=process.default_chat_template,
                    chat_template=DEFAULT_CHAT_TEMPLATE,
                    add_generation_prompt=True,
                    tokenize=False)

audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                audios.append(
                    librosa.load(BytesIO(urlopen(ele['audio_url']).read()),
                                    sr=feature_extractor.sampling_rate)[0])

inputs = tokenizer(text, padding=True, return_tensors="pt")
# ipdb.set_trace()
audio_inputs = feature_extractor(audios,
                                 sampling_rate=feature_extractor.sampling_rate,
                                 return_attention_mask=True,
                                 padding="max_length",
                                 return_tensors="pt")
# ipdb.set_trace()

audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
inputs.update(audio_inputs)
model_inputs = BatchFeature(data={**inputs}).to('mps')

# model_config = Qwen2AudioConfig.from_pretrained(QWEN2_AUDIO_MODEL_PATH)
# model_encoder = Qwen2AudioEncoder._from_config(model_config.audio_config)
# multi_modal_projector = Qwen2AudioMultiModalProjector(model_config)

model = Qwen2AudioForConditionalGeneration.from_pretrained(QWEN2_AUDIO_MODEL_PATH, torch_dtype=torch.float16)
model = model.to('mps')
generate_ids = model.generate(**model_inputs, max_length=512)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]

response = tokenizer.batch_decode(
                        generate_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)[0]
print(response)
