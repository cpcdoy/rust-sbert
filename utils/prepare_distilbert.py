import os
import numpy as np
import torch
import subprocess
from pathlib import Path

def convert_to_c_array(target_path, prefix="", suffix=False):
    config_path = str(target_path + '/config.json')
    vocab_path = str(target_path + '/vocab.txt')
    model_path = str(target_path + '/pytorch_model.bin')

    weights = torch.load(model_path, map_location='cpu')
    nps = {}
    for k, v in weights.items():
        k_distil = prefix + k
        if suffix:
            k_distil = k_distil.split('.')[-1]
        print(k_distil)
        nps[k_distil] = np.ascontiguousarray(v.cpu().numpy())

    np.savez(target_path + '/model.npz', **nps)

    source = str(target_path + '/model.npz')
    target = str(target_path + '/model.ot')

    toml_location = (Path(__file__).resolve() / '..' / '..' / 'Cargo.toml').resolve()
    
    subprocess.call(
        ['cargo', 'run', '--bin=convert-tensor', '--manifest-path=%s' % toml_location, '--', source, target])

convert_to_c_array('./0_DistilBERT', prefix='distilbert.')
convert_to_c_array('./2_Dense', suffix=True)