import ipdb
import json
from pathlib import Path, PosixPath
from typing import Dict
from tokenizers import AddedToken, Tokenizer
import mlx.core as mx
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
    model_input_names = ["input_ids", "attention_mask"]
    truncation_side = "right"

    def __init__(self, model_path: str):
        resolved_vocab_files = {
            # 'vocab_file': check_file(Path(model_path)/'vocab.json'),
            # 'merges_file': check_file(Path(model_path)/'merges.txt'),
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
        self.bos_token = config_kwargs['bos_token']  # -> None
        self.eos_token = config_kwargs['eos_token']  # -> AddedToken()
        self.unk_token = config_kwargs['unk_token']  # -> None
        self.pad_token = config_kwargs['pad_token']  # -> AddedToken()
        self.additional_special_tokens = config_kwargs['additional_special_tokens']  # -> list

        # from class PretrainedTokenizerFast
        self.tokenizer = Tokenizer.from_file(str(config_kwargs["tokenizer_file"]))
        self.tokenizer.no_truncation()

        # from class PreTrainedTokenizerBase
        self.model_max_length = config_kwargs["model_max_length"]  # 8192
        self.padding_side = config_kwargs["padding_side"]  # left

        self.truncation_side = config_kwargs.pop("truncation_side", self.truncation_side)
        self.clean_up_tokenization_spaces = config_kwargs["clean_up_tokenization_spaces"]  # False
        self.split_special_tokens = config_kwargs["split_special_tokens"]  # False
        self.chat_template = config_kwargs["chat_template"]  # str

        self.tokenizer.encode_special_tokens = self.split_special_tokens

        added_tokens_decoder_hash = {hash(repr(token)) for token in self.tokenizer.get_added_tokens_decoder()}
        tokens_to_add = [
            token for index, token in
            sorted(added_tokens_decoder.items(), key=lambda x: x[0])
            if hash(repr(token)) not in added_tokens_decoder_hash
        ]

        added_tokens_encoder = {
            k.content: v for v, k in
            sorted(self.tokenizer.get_added_tokens_decoder().items(), key=lambda item: item[0])
        }
        encoder = [added_tokens_encoder.keys()] + [str(token) for token in tokens_to_add]
        # ipdb.set_trace()

        tokens_to_add += [
            token for token in
            self.all_special_tokens_extended
            if token not in encoder and token not in tokens_to_add
        ]

        special_tokens = [str(s) for s in self.all_special_tokens_extended]
        is_last_special = None
        tokens = []
        for token in tokens_to_add:
            is_special = (
                (token.special or str(token) in special_tokens)
                if isinstance(token, AddedToken)
                else str(token) in special_tokens
            )
            if is_last_special is None or is_last_special == is_special:
                tokens.append(token)
            else:
                self._add_tokens(tokens, special_tokens=is_last_special)
                tokens = [token]
            is_last_special = is_special
        if tokens:
            self._add_tokens(tokens, special_tokens=is_last_special)

    @property
    def special_tokens_map_extended(self):
        set_attr = {}
        set_attr['eos_token'] = self.eos_token
        set_attr['pad_token'] = self.pad_token
        set_attr['additional_special_tokens'] = self.additional_special_tokens
        return set_attr

    @property
    def all_special_tokens_extended(self):
        all_tokens = []
        seen = set()
        for value in self.special_tokens_map_extended.values():
            if isinstance(value, (list, tuple)):
                tokens_to_add = [token for token in value if str(token) not in seen]
            else:
                tokens_to_add = [value] if str(value) not in seen else []
            seen.update(map(str, tokens_to_add))
            all_tokens.extend(tokens_to_add)
        return all_tokens

    def _add_tokens(self, new_tokens, special_tokens=False):
        if special_tokens:
            return self.tokenizer.add_special_tokens(new_tokens)

        return self.tokenizer.add_tokens(new_tokens)

    def __call__(self, text: str, padding: bool = True):
        encodings = self.tokenizer.encode_batch([text],
                                                add_special_tokens=True,
                                                is_pretokenized=False)[0]
        encoding_dict = {}
        encoding_dict['input_ids'] = mx.array(encodings.ids)
        encoding_dict["attention_mask"] = mx.array(encodings.attention_mask)

        return encoding_dict
