import sys
import os
import numpy as np
import torch
import subprocess
from pathlib import Path

def convert_to_c_array(target_path):
    model_path = str(target_path + '/pytorch_model.bin')

    weights = torch.load(model_path, map_location='cpu')
    nps = {}
    for k, v in weights.items():
        k = k.replace("gamma", "weight").replace("beta", "bias")
        print(k)
        nps[k] = np.ascontiguousarray(v.cpu().numpy())

    np.savez(target_path + '/model.npz', **nps)

    source = str(target_path + '/model.npz')
    target = str(target_path + '/model.ot')

    toml_location = (Path(__file__).resolve() / '..' / '..' / 'Cargo.toml').resolve()

    subprocess.call(
        ['cargo', 'run', '--bin=convert-tensor', '--manifest-path=%s' % toml_location, '--', source, target])

if __name__ == "__main__":
    root = sys.argv[1]
    convert_to_c_array(root)
