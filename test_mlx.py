from mlxs2s.tokenizer import MLXQwen2Tokenizer
from mlxs2s.feature_extractor import MLXQwen2FeatureExtractor
from mlxs2s.text import TEXT
import librosa

QWEN2_AUDIO_MODEL_PATH = '/Users/liwei15/huggingface_models/Qwen2-Audio-7B-Instruct/'

# mlx_tokenizer = MLXQwen2Tokenizer(QWEN2_AUDIO_MODEL_PATH)
# mlx_input = mlx_tokenizer(TEXT)

conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {"role": "user", "content": [
        {"type": "text", "text": "Stay alert and cautious, What does the person say?"},
        # {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"},
        {"type": "audio", "audio_url": "1272-128104-0000.flac"},
    ]},
]

mlx_feature_extractor = MLXQwen2FeatureExtractor(QWEN2_AUDIO_MODEL_PATH)

audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                audios.append(
                    librosa.load(ele['audio_url'], sr=mlx_feature_extractor.sampling_rate)[0]
                )

out = mlx_feature_extractor(audios)
