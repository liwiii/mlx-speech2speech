import torch
from safetensors.torch import save_file
# from copy import deepcopy

cosyvoice_llm_torch_path = '/Users/liwei15/repo/CosyVoice/pretrained_models/CosyVoice-300M/llm.pt'

# st -> safetensors
cosyvoice_llm_st_f32_path = cosyvoice_llm_torch_path.replace('.pt', '_f32.safetensors')
cosyvoice_llm_st_bf16_path = cosyvoice_llm_torch_path.replace('.pt', '_bf16.safetensors')

tensors = torch.load(cosyvoice_llm_torch_path, map_location='cpu')

# tensors_bf16 = deepcopy(tensors)
save_file(tensors, cosyvoice_llm_st_f32_path)

for key in tensors:
    if 'norm' not in key:
        tensors[key] = tensors[key].bfloat16()

save_file(tensors, cosyvoice_llm_st_bf16_path)
