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
    return modules, hparams, device

def count_scp_lines(scpfile):
    lines = 0
    with open(scpfile) as fin:
        for _ in fin:
            lines += 1
    return lines

def run_test(modules, hparams, device):
    prior = torch.load(hparams.prior_file).to(device)
    num_utts = count_scp_lines(hparams.test_feats)
    with open(hparams.test_probs_out, 'wb') as fo:
        with torch.no_grad():
            for uttid, feats in tqdm.tqdm(kaldi_io.read_mat_scp(hparams.test_feats), total=num_utts):
                feats = torch.from_numpy(feats)
                padded_feats = torch.cat(
                        (
                            feats[0].unsqueeze(0).repeat_interleave(hparams.contextlen,dim=0),
                            feats,
                            feats[-1].unsqueeze(0).repeat_interleave(hparams.contextlen,dim=0)
                        )
                )
                padded_feats = padded_feats.unsqueeze(0).to(device)
                normalized = modules.normalize(padded_feats, lengths=torch.tensor([1.]))
                encoded_all = modules.encoder(normalized)
                front_index = hparams.front_index
                back_index = front_index + feats.shape[0] // hparams.subsampling
                encoded_relevant = encoded_all[:,front_index:back_index,:]
                out = modules.lin_out(encoded_relevant)
                predictions = hparams.log_softmax(out)
                normalized_predictions = predictions - prior
                kaldi_io.write_mat(fo, normalized_predictions.squeeze(0).cpu().numpy(), key=uttid)
    

if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    modules, hparams, device = setup(hparams, run_opts)
    run_test(modules, hparams, device)
