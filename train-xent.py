#!/usr/bin/env python3

import sys
import speechbrain as sb
import webdataset as wds
from hyperpyyaml import load_hyperpyyaml


class XentBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        #print("Audio shape:", batch.sig.data.shape)
        #feats = self.hparams.feat_maker(batch.sig.data)
        print("Feature shape:", batch.feats.data.shape)
        print("Alignment shape:", batch.ali.data.shape)
        print("Feature time float division by three:", batch.feats.data.shape[1]/3)
        print("Feature time int division by three:", batch.feats.data.shape[1]//3)
        print("Alignment shape multiplied by three:", batch.ali.data.shape[1]*3)
        sys.exit(1)

    def compute_objectives(self, predictions, batch, stage):
        pass


def make_datasets(hparams):
    valid_data = (
            wds.WebDataset(hparams["valid_shards"])
            .decode()
            .rename(
                #text="transcript.txt",
                ali="ali.pth",
                feats="feats.pth",
                #sig="audio.pth",
                #meta="meta.json"
            )  
            .batched(
                batchsize=2,
                collation_fn=sb.dataio.batch.PaddedBatch
            )
    )
    return {"train": valid_data, "valid": valid_data, "test": valid_data}

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fi:
        hparams = load_hyperpyyaml(fi, overrides)
    sb.create_experiment_directory(
            experiment_directory = hparams["expdir"],
            hyperparams_to_save = hparams_file,
            overrides=overrides 
   
    )
    datasets = make_datasets(hparams)
    brain = XentBrain(
            modules = hparams["modules"],
            opt_class = hparams["opt_class"],
            hparams = hparams,
            run_opts = run_opts,
            #checkpointer = hparams["checkpointer"]
    )
    brain.fit(
            range(10),
            datasets["train"],
            datasets["valid"],
            train_loader_kwargs=hparams["train_loader_kwargs"]
    )
