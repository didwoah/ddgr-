
import os
import torch

def add_mode(cmd, mode):
    return cmd + f" --{mode}"
def add_param(cmd, name, value):
    return cmd + f" --{name} {value}"


base_cmd = "python main.py"

base_cmd = add_mode(base_cmd, "ddim")
base_cmd = add_mode(base_cmd, "pre_train")
base_cmd = add_mode(base_cmd, "dual_guidance")

cfg_factors = [1.8]
cg_factors = torch.linspace(start=0.1, end=1, steps=5).tolist()

print(f"cfg_factors: {cfg_factors}")
print(f"cg_factors: {cg_factors}")

for cg_factor in cg_factors:
    for cfg_factor in cfg_factors:
        cmd = add_param(base_cmd, "cfg_factor", cfg_factor)
        cmd = add_param(cmd, "cg_factor", cg_factor)

        print(f"command: {cmd}")
        os.system(cmd)
