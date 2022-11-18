#!/usr/bin/env/python3
"""Finnish Parliament ASR
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import webdataset as wds
from glob import glob
import io
import torchaudio
import local
import tqdm
from pychain import ChainGraph, ChainGraphBatch 
import simplefst
import pathlib

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class LFMMIAM(sb.Brain):
    def __init__(self, train_fsts={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_fsts = train_fsts

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        feats = (self.hparams.compute_features(batch.wav.data)).detach()
        normalized = self.modules.normalize(feats, lengths=batch.wav.lengths)
        encoded = self.modules.encoder(normalized)
        out = self.modules.lin_out(encoded)
        return out 

    def compute_objectives(self, predictions, batch, stage):
        num_transitions = list(map(self.hparams.transgetter, batch.graph))
        output_lengths = (predictions.shape[1] * batch.wav.lengths).int().cpu()
        max_num_states = max(map(self.hparams.stategetter, batch.graph))
        numerator_graphs = ChainGraphBatch(
                batch.graph,
                max_num_transitions=max(num_transitions),
                max_num_states=max_num_states
        )
        chain_loss = self.hparams.chain_loss(predictions, output_lengths, numerator_graphs)
        output_norm_loss = torch.linalg.norm(predictions,dim=2).mean()
        loss = chain_loss + 0.0005 * output_norm_loss
        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
                num_to_keep=getattr(self.hparams, "ckpts_to_keep", 1)
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        if getattr(self.hparams, "avg_ckpts", 1) > 1:
            ckpts = self.checkpointer.find_checkpoints(
                    max_key=max_key,
                    min_key=min_key,
                    max_num_checkpoints=self.hparams.avg_ckpts
            )
            model_state_dict = sb.utils.checkpoints.average_checkpoints(
                    ckpts, "model" 
            )
            self.hparams.model.load_state_dict(model_state_dict)
            self.checkpointer.save_checkpoint(name=f"AVERAGED-{self.hparams.avg_ckpts}")

def numfsts_to_local_tmp(fstdir, tmpdir):
    """Copies the chain numerator FSTs onto a local disk"""
    fstdir = pathlib.Path(fstdir)
    tmpdir = pathlib.Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    sb.utils.superpowers.run_shell(f"rsync --update {fstdir}/num.*.ark {tmpdir}/")
    numfsts = {}
    for scpfile in fstdir.glob("num.*.scp"):
        with open(scpfile) as fin:
            for line in fin:
                uttid, data = line.strip().split()
                arkpath, offset = data.split(":")
                newpath = arkpath.replace(str(fstdir), str(tmpdir))
                numfsts[uttid] = (newpath, int(offset))
    return numfsts

def dataio_prepare(hparams, numfsts):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys mapping to 
        WebDataset datasets dataloaders for them.
    """

    def load_valid_fst(sample):
        uttid = sample["__key__"]
        fstpath, offset = numfsts["valid"][uttid]
        sample["graph"] = ChainGraph(simplefst.StdVectorFst.read_ark(fstpath, offset), log_domain=True)
        return sample

    traindata = (
            wds.WebDataset(hparams["trainshards"])
            .decode()
            .rename(wav="audio.pth")
            .repeat()
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                **hparams["dynamic_batch_kwargs"]
            )
    )
    validdata = (
            wds.WebDataset(hparams["validshards"])
            .decode()
            .rename(wav="audio.pth")
            .map(load_valid_fst, handler=wds.warn_and_continue)
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                drop_end=False,
                **hparams["valid_dynamic_batch_kwargs"],
            )
    )
    return {"train": traindata, "valid": validdata}




if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Copy numerator FSTs to local drive:
    numfsts = {}
    numfsts["train"] = numfsts_to_local_tmp(hparams["numfstdir"], hparams["numfsttmpdir"])
    numfsts["valid"] = numfsts_to_local_tmp(hparams["valid_numfstdir"], hparams["valid_numfsttmpdir"])

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams, numfsts)
    # read valid data into memory:
    datasets["valid"] = torch.utils.data.DataLoader(
            list(iter(datasets["valid"])),
            batch_size=None
    )
    # Then we can copy the train FSTs into memory:
    TRAIN_FSTS = {}
    print("Reading training FSTs to memory")
    for uttid, (fstpath, offset) in tqdm.tqdm(numfsts["train"].items()):
        TRAIN_FSTS[uttid] = ChainGraph(simplefst.StdVectorFst.read_ark(fstpath, offset), log_domain=True)

    # Pretrain if defined:
    if "pretrainer" in hparams:
        if "pretrain_max_key" in hparams:
            ckpt = hparams["ckpt_finder"].find_checkpoint(max_key=hparams["pretrain_max_key"])
        elif "pretrain_min_key" in hparams:
            ckpt = hparams["ckpt_finder"].find_checkpoint(min_key=hparams["pretrain_min_key"])
        else:
            ckpt = hparams["ckpt_finder"].find_checkpoint()
        hparams["pretrainer"].collect_files(ckpt.path)
        hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = LFMMIAM(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        train_fsts = TRAIN_FSTS,
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs = hparams["train_loader_kwargs"]
    )
