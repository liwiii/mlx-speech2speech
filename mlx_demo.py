# import ipdb
# import time
import sys
import numpy as np
import array
import wave
from mlxs2s.tokenizer import MLXQwen2Tokenizer
from mlxs2s.feature_extractor import MLXQwen2FeatureExtractor
from mlxs2s.models import Qwen2AudioForConditionalGeneration
from mlxs2s.text import TEXT, TEXT2, TEXT3
# import librosa
import mlx.core as mx
import pyaudio
from alive_progress import alive_bar

QWEN2_AUDIO_MODEL_PATH = '/Users/liwei15/huggingface_models/Qwen2-Audio-7B-Instruct/'

CHUNK = 1600
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

mlx_tokenizer = MLXQwen2Tokenizer(QWEN2_AUDIO_MODEL_PATH)
mlx_feature_extractor = MLXQwen2FeatureExtractor(QWEN2_AUDIO_MODEL_PATH)
model = Qwen2AudioForConditionalGeneration(QWEN2_AUDIO_MODEL_PATH)

input_ids = mlx_tokenizer(TEXT2)
# mx.async_eval(input_ids)
mx.eval(input_ids)

print("A very simple speech translation demo based on Qwen2Audio")
while True:
    user_input = input((
        " * press 's' to start recording and recognize,\n"
        " * press 'q' to quit\n"
    ))
    if user_input == 's':
        audio_data = []
        wf = wave.open('debug.wav', 'wb')
        p = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)
        print('Recording...')
        with alive_bar(RECORD_SECONDS) as bar:
            for ind in range(0, RATE // CHUNK * RECORD_SECONDS):
                audio_data += array.array('f', stream.read(CHUNK))
                if ind % 10 == 0:
                    bar()
        stream.close()
        p.terminate()

        audio = np.array(audio_data, dtype=np.float32)
        wf.writeframes(audio.tobytes())
        wf.close()

        print('Recording Done as Replay:')
        p = pyaudio.PyAudio()

        # Open stream (2)
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True)

        # Play samples from the wave file (3)
        stream.write(audio.tobytes())
        # Close stream (4)
        stream.close()
        # Release PortAudio system resources (5)
        p.terminate()

        audios = [audio]

        audios[0] = np.pad(audios[0], [200, 200], mode='reflect')  # in order to be compatible with torch.stft() with center=True
        # mlx_input["input_features"] = mlx_feature_extractor(audios).T[None,]  # [1,feature_dim, feature_num]

        # [1, ] is necessary
        print("Analysis feature...")
        input_features = mlx_feature_extractor(audios)  # [1, feature_num, feature_dim]
        mx.async_eval(input_features)

        print("Start decoding...", flush=True)
        output_tokens = model(input_ids=input_ids, input_features=input_features, max_decoder_tokens=256)
        # print(output_tokens, flush=True)
        out_str = mlx_tokenizer.tokenizer.decode(
                    output_tokens,
                    skip_special_tokens=True)
        print("RESUT:")
        print(f"\t{out_str}\n\n")
    elif user_input == 'q':
        print("Quit, Byt bye~")
        sys.exit(0)
    else:
        print(f"ILLEGAL INPUT: {user_input}, Only 'q' or 's' supported now")

        


