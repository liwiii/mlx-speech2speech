# import ipdb
import json
from pathlib import Path, PosixPath
from typing import Dict
from tokenizers import AddedToken, Tokenizer
# from transformers import Qwen2TokenizerFast


def check_file(file_path: PosixPath):
    if not file_path.is_file():
        raise FileNotFoundError(f"Can not find necessary file: {file_path}")
    else:
        return file_path


class MLXQwen2Tokenizer:
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]

    def __init__(self, model_path: str):
        resolved_vocab_files = {
            'vocab_file': check_file(Path(model_path)/'vocab.json'),
            'merges_file': check_file(Path(model_path)/'merges.txt'),
            'tokenizer_file': check_file(Path(model_path)/'tokenizer.json'),
            'tokenizer_config_file': check_file(Path(model_path)/'tokenizer_config.json')
        }
        with open(resolved_vocab_files['tokenizer_config_file'], encoding="utf-8") as handle:
            config_kwargs = json.load(handle)
        config_kwargs.update(resolved_vocab_files)

        added_tokens_decoder: Dict[int, AddedToken] = {}
        added_tokens_map: Dict[str, AddedToken] = {}
        for idx, token in config_kwargs["added_tokens_decoder"].items():
            if isinstance(token, dict):
                token = AddedToken(**token)
            if isinstance(token, AddedToken):
                added_tokens_decoder[int(idx)] = token
                added_tokens_map[str(token)] = token
            else:
                raise ValueError(
                    f"Found a {token.__class__} in the saved `added_tokens_decoder`, should be a dictionary or an AddedToken instance")

        config_kwargs["added_tokens_decoder"] = added_tokens_decoder
        for key in self.SPECIAL_TOKENS_ATTRIBUTES & config_kwargs.keys():
            # {'unk_token', 'additional_special_tokens', 'pad_token', 'bos_token', 'eos_token'}
            if key != "additional_special_tokens":
                config_kwargs[key] = added_tokens_map.get(
                    str(config_kwargs[key]), config_kwargs[key])

        # ipdb.set_trace()
        self.bos_token = config_kwargs.pop('bos_token')  # -> None
        self.eos_token = config_kwargs.pop('eos_token')  # -> AddedToken()
        self.unk_token = config_kwargs.pop('unk_token')  # -> None
        self.pad_token = config_kwargs.pop('pad_token')  # -> AddedToken()
        self.additional_special_tokens = config_kwargs.pop(
            'additional_special_tokens')  # -> list

        # from class PretrainedTokenizerFast
        self.tokenizer = Tokenizer.from_file(
            str(config_kwargs.pop("tokenizer_file")))
        self.tokenizer.no_truncation()

        # from class PreTrainedTokenizerBase
        model_max_length = config_kwargs.pop("model_max_length")  # 8192
        self.model_max_length = model_max_length
        self.padding_side = config_kwargs.pop("padding_side")

        self.model_input_names = ['input_ids', 'attention_mask']  # hardcode

        self.clean_up_tokenization_spaces = config_kwargs.pop(
            "clean_up_tokenization_spaces")  # False
        self.split_special_tokens = config_kwargs.pop(
            "split_special_tokens")  # False
        self.chat_template = config_kwargs.pop("chat_template")  # str

    def __call__(self, text: str, padding: bool = True):
        encodings = self.tokenizer.encode_batch([text],
                                                add_special_tokens=True,
                                                is_pretokenized=False)[0]
        encoding_dict = {}
        encoding_dict['input_ids'] = encodings.ids
        encoding_dict["attention_mask"] = encodings.attention_mask

        return encoding_dict

