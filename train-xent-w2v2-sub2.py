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
class XentAM(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        feats = self.modules.wav2vec2(batch.wav.data)
        encoded = self.modules.enc(feats)
        xent_out = self.modules.xent_lin_out(encoded)
        xent_predictions = self.hparams.log_softmax(xent_out)
        return xent_predictions

    def compute_objectives(self, predictions, batch, stage):
        loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions,
            length=batch.ali.lengths,
            targets=batch.ali.data,
            label_smoothing=self.hparams.label_smoothing,
        )
        if stage != sb.Stage.TRAIN:
            min_length = min(predictions.shape[1], batch.ali.data.shape[1])
            self.accuracy_metric.append(predictions[:,:min_length,:], batch.ali.data[:,:min_length], length=batch.ali.lengths)
        return loss

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.accuracy_metric = self.hparams.accuracy_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["accuracy"] = self.accuracy_metric.summarize()

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            if not self.hparams.wav2vec2.freeze:
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr_wav2vec
                )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"], "xent-accuracy": stage_stats["accuracy"]}, 
                min_keys=["loss"],
                num_to_keep=getattr(self.hparams, "ckpts_to_keep", 1)
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        if not self.hparams.wav2vec2.freeze:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "wav2vec_opt", self.wav2vec_optimizer
                )
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    
    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        should_step = self.step % self.hparams.grad_accumulation_factor == 0
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        (loss / self.hparams.grad_accumulation_factor).backward()

        if should_step:
            if self.check_gradients(loss):
                if not self.hparams.wav2vec2.freeze:
                    self.wav2vec_optimizer.step()
                self.model_optimizer.step()
        
            if not self.hparams.wav2vec2.freeze:
                self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()

        return loss.detach()


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


def dataio_prepare(hparams):
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
    traindata = (
            wds.WebDataset(hparams["trainshards"])
            .decode()
            .rename(wav="audio.pth", ali="ali.pth")
            .repeat()
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                **hparams["dynamic_batch_kwargs"]
            )
    )
    validdata = (
            wds.WebDataset(hparams["validshards"])
            .decode()
            .rename(wav="audio.pth", ali="ali.pth")
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                **hparams["valid_dynamic_batch_kwargs"],
                drop_end=False
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

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    # Pretrain if defined:
    if "pretrainer" in hparams:
        ckpt = hparams["ckpt_finder"].find_checkpoint(max_key="accuracy")
        hparams["pretrainer"].collect_files(ckpt.path)
        hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = XentAM(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
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
    prior = asr_brain.estimate_prior_empirical(
            datasets["train"], 
            loader_kwargs=hparams["prior_loader_kwargs"],
            max_key=hparams["test_max_key"]
    )
    torch.save(prior, hparams["prior_file"])
