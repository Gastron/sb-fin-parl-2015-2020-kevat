#!/usr/bin/env/python3
"""Finnish Parliament ASR"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import kaldi_io
import tqdm
from types import SimpleNamespace
sys.path.append("./local")
from make_shards import wavscp_to_output

def setup(hparams, run_opts):
    """ Kind of mimics what Brain does """
    if "device" in run_opts:
        device = run_opts["device"]
    elif "device" in hparams:
        device = hparams["device"]
    else:
        device = "cpu"
    print("Device is:", device)
    if "cuda" in device:
        torch.cuda.set_device(int(device[-1]))
    modules = torch.nn.ModuleDict(hparams["modules"]).to(device)
    hparams = SimpleNamespace(**hparams)
    if hasattr(hparams, "checkpointer"):
        if hasattr(hparams, "test_max_key"):
            ckpt = hparams.checkpointer.find_checkpoint(max_key=hparams.test_max_key)
        elif hasattr(hparams, "test_min_key"):
            ckpt = hparams.checkpointer.find_checkpoint(min_key=hparams.test_min_key)
        else:
            ckpt = hparams.checkpointer.find_checkpoint()
        hparams.checkpointer.load_checkpoint(ckpt)
        epoch = hparams.epoch_counter.current
        print("Loaded checkpoint from epoch", epoch, "at path", ckpt.path)
    modules.eval()
    return modules, hparams, device

def count_scp_lines(scpfile):
    lines = 0
    with open(scpfile) as fin:
        for _ in fin:
            lines += 1
    return lines

def run_test(modules, hparams, device):
    num_utts = count_scp_lines(hparams.test_wavscp)
    lin_out_name = getattr(hparams, "lin_out_name", "lfmmi_lin_out")
    lin_out = getattr(modules, lin_out_name)
    with open(hparams.test_probs_out, 'wb') as fo:
        data_iter = wavscp_to_output(hparams.test_wavscp)
        with torch.no_grad():
            for uttid, data in tqdm.tqdm(data_iter, total=num_utts):
                audio = data["audio.pth"].to(device).unsqueeze(0)
                feats = hparams.compute_features(audio)
                normalized = modules.normalize(feats, lengths=torch.tensor([1.]), epoch=1000)
                encoded = modules.encoder(normalized)
                out = lin_out(encoded)
                kaldi_io.write_mat(fo, out.squeeze(0).cpu().numpy(), key=uttid)
    

if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    modules, hparams, device = setup(hparams, run_opts)
    run_test(modules, hparams, device)
