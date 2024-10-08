TEXT = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Stay alert and cautious, What does the person say? Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>
<|im_end|>
<|im_start|>assistant
'''

TEXT2 = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Stay alert and cautious, recognize the speech from audio. Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>
<|im_end|>
<|im_start|>assistant
'''

TEXT3 = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Stay alert and cautious, recognize the speech from audio and translate into Japanese. Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>
<|im_end|>
<|im_start|>assistant
'''

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
