import sys
import json
from pathlib import Path
import numpy as np
from mlxs2s.utils import check_file
import mlx.core as mx


def mel_to_hertz(mels, mel_scale: str = "htk"):
    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    if mel_scale == "htk":
        return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
    elif mel_scale == "kaldi":
        return 700.0 * (np.exp(mels / 1127.0) - 1.0)

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0

    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))

    return freq


def hertz_to_mel(freq, mel_scale: str = "htk"):
    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))
    elif mel_scale == "kaldi":
        return 1127.0 * np.log(1.0 + (freq / 700.0))

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0

    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep

    return mels


def _create_triangular_filter_bank(fft_freqs: np.ndarray, filter_freqs: np.ndarray):
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
    norm=None,
    mel_scale: str = "htk",
    triangularize_in_mel_space: bool = False,
):
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    # center points of the triangular mel filters
    mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)

    if triangularize_in_mel_space:
        # frequencies of FFT bins in Hz, but filters triangularized in mel space
        fft_bin_width = sampling_rate / (num_frequency_bins * 2)
        fft_freqs = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale)
        filter_freqs = mel_freqs
    else:
        # frequencies of FFT bins in Hz
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        mel_filters *= np.expand_dims(enorm, 0)

    if (mel_filters.max(axis=0) == 0.0).any():
        sys.stderr.write(
            "At least one mel filter has all zero values. "
            f"The value for `num_mel_filters` ({num_mel_filters}) may be set too high. "
            f"Or, the value for `num_frequency_bins` ({num_frequency_bins}) may be set too low."
        )

    return mel_filters


class MLXQwen2FeatureExtractor:
    def __init__(self, model_path: str):
        preprocessor_config_file = check_file(Path(model_path)/'preprocessor_config.json')
        with open(preprocessor_config_file, encoding="utf-8") as handle:
            config_kwargs = json.load(handle)

        self.feature_size = config_kwargs["feature_size"]  # 128
        self.sampling_rate = config_kwargs["sampling_rate"]  # 16000
        self.padding_value = config_kwargs["padding_value"]  # 0.0
        self.return_attention_mask = config_kwargs["return_attention_mask"]  # true
        self.n_fft = config_kwargs["n_fft"]  # 400
        self.hop_length = config_kwargs["hop_length"]  # 160
        self.chunk_length = config_kwargs["chunk_length"]  # 30
        self.n_samples = self.chunk_length * self.sampling_rate  # 480000
        self.nb_max_frames = self.n_samples // self.hop_length  # 100

        self.mel_filters = mx.array(mel_filter_bank(
            num_frequency_bins=1 + self.n_fft // 2,
            num_mel_filters=self.feature_size,
            min_frequency=0.0,
            max_frequency=self.sampling_rate/2,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        ))
        self.m_window = mx.array(np.hanning(self.n_fft + 1)[:-1])
        # self.t_window = torch.hann_window(self.n_fft)
    
    '''
    def _torch_extract_fbank_features(self, waveform: np.array, device: str = "cpu") -> np.ndarray:
        waveform = torch.from_numpy(waveform).type(torch.float32)

        window = torch.hann_window(self.n_fft)
        if device != "cpu":
            waveform = waveform.to(device)
            window = window.to(device)
        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, center=False, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
        if device != "cpu":
            mel_filters = mel_filters.to(device)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        if device != "cpu":
            log_spec = log_spec.detach().cpu()
        return log_spec.numpy()
    '''

    def _mlx_extract_fbank_features(self, waveform, device=mx.cpu):
        with mx.stream(device):
            waveform = mx.array(waveform)
            shape = [(waveform.shape[0]-self.n_fft+self.hop_length)//self.hop_length, self.n_fft]
            strides = [self.hop_length, 1]
            stft = mx.fft.rfft(mx.as_strided(waveform, shape=shape, strides=strides) * self.m_window)
            magnitudes = stft.abs().square()
            mel_spec = magnitudes @ self.mel_filters
            log_spec = mx.clip(mel_spec, a_min=1e-10, a_max=None).log10()
            log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            mx.eval(log_spec)
        return log_spec

    def __call__(self, raw_speech):
        return self._mlx_extract_fbank_features(raw_speech[0])[None,].astype(mx.bfloat16)
