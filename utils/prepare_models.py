import sys
import os
import numpy as np
import torch
import subprocess
from pathlib import Path

import requests
import zipfile
import shutil

def download_model(url, filename):
    print("Downloading model...")

    r = requests.get(url, allow_redirects=True)

    zip_filename = filename + ".zip"
    open(zip_filename, 'wb').write(r.content)

    model_dir = str("models/") + filename
    os.makedirs(model_dir, exist_ok=True)

    print("Extracting model...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

    shutil.move(model_dir + "/0_Transformer", model_dir + "/0_DistilBERT")

    os.remove(zip_filename)

    print("Done.")

    return model_dir

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

if __name__ == "__main__":
    url = "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/distiluse-base-multilingual-cased-v1.zip"
    
    path = download_model(url, "distiluse-base-multilingual-cased")
    
    convert_to_c_array(path + '/0_DistilBERT', prefix='distilbert.')
    convert_to_c_array(path + '/2_Dense', suffix=True)