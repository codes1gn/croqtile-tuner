import torch
import numpy as np

path = "/home/baldlee/workspace/choreo-attn/gdn/reference/test_output_bf16.pt"
data = torch.load(path)
o_bf16 = data["o_bf16"]

o_uint16 = o_bf16.view(torch.uint16)
flat = o_uint16.flatten()

output_path = "/home/baldlee/workspace/choreo-attn/gdn/reference/test_output_bf16.bin"
flat.cpu().numpy().tofile(output_path)
print(f"Saved {flat.numel()} uint16_t values to {output_path}")
print(f"Shape: {o_bf16.shape}")
