import ipdb
import numpy as np
from mlxs2s.tokenizer import MLXQwen2Tokenizer
from mlxs2s.feature_extractor import MLXQwen2FeatureExtractor
from mlxs2s.models import Qwen2AudioForConditionalGeneration
from mlxs2s.text import TEXT
import librosa
import mlx.core as mx

QWEN2_AUDIO_MODEL_PATH = '/Users/liwei15/huggingface_models/Qwen2-Audio-7B-Instruct/'

conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {"role": "user", "content": [
        {"type": "text", "text": "Stay alert and cautious, What does the person say?"},
        # {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"},
        {"type": "audio", "audio_url": "1272-128104-0000.flac"},
    ]},
]

mlx_tokenizer = MLXQwen2Tokenizer(QWEN2_AUDIO_MODEL_PATH)
mlx_feature_extractor = MLXQwen2FeatureExtractor(QWEN2_AUDIO_MODEL_PATH)
model = Qwen2AudioForConditionalGeneration(QWEN2_AUDIO_MODEL_PATH)
model.eval()


mlx_input = mlx_tokenizer(TEXT)
# ipdb.set_trace()
audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                audios.append(
                    librosa.load(ele['audio_url'], sr=mlx_feature_extractor.sampling_rate)[0]
                )

audios[0] = np.pad(audios[0], [200, 200], mode='reflect')  # in order to be compatible with torch.stft() with center=True
# mlx_input["input_features"] = mlx_feature_extractor(audios).T[None,]  # [1,feature_dim, feature_num]

# [1, ] is necessary
mlx_input["input_features"] = mlx_feature_extractor(audios)[None,].astype(mx.bfloat16)  # [1, feature_num, feature_dim]
mlx_input["feature_attention_mask"] = mx.ones([mlx_input["input_features"].shape[2]], dtype=mx.int32)

out = model(**mlx_input)
