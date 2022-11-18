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
from ptpython.repl import embed


logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class XentAM(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        feats = (self.hparams.compute_features(batch.wav.data)).detach()
        print(feats.shape[1], batch.ali.data.shape[1])
        epoch = self.hparams.epoch_counter.current
        normalized = self.modules.normalize(feats, lengths=batch.wav.lengths, epoch=epoch)
        encoded = self.modules.encoder(normalized, lengths=batch.wav.lengths)
        out = self.modules.lin_out(encoded)
        predictions = self.hparams.log_softmax(out)
        return predictions

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

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["accuracy"] = self.accuracy_metric.summarize()

        old_lr = self.hparams.lr_annealing.current_lr

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"accuracy": stage_stats["accuracy"]}, max_keys=["accuracy"],
                num_to_keep=getattr(self.hparams, "ckpts_to_keep", 1)
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            if self.check_gradients(loss):
                old_lr, new_lr = self.hparams.lr_annealing(self.optimizer)
                self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.detach().cpu()


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

    def estimate_prior_empirical(self, train_data, loader_kwargs={}, max_key=None, min_key=None):
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.hparams.train_logger.log_stats(
            stats_meta={"Epoch loaded for prior": self.hparams.epoch_counter.current},
        )
        dataloader = self.make_dataloader(train_data, **loader_kwargs, stage=sb.Stage.TEST)
        with torch.no_grad():
            prior_floor = 1.0e-15
            prior = torch.ones((self.hparams.num_units,)) * prior_floor
            for batch in tqdm.tqdm(dataloader):
                log_predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
                predictions = log_predictions.exp()
                lengths = batch.wav.lengths*predictions.shape[1]
                mask = sb.dataio.dataio.length_to_mask(lengths).float()
                summed_preds = torch.sum(predictions * mask.unsqueeze(-1), dim=(0,1))
                prior += summed_preds.detach().cpu()
            # Normalize:
            prior = prior / prior.sum()
        return prior.log()

    def estimate_prior_frequency(self, train_data, loader_kwargs={}, max_key=None, min_key=None):
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.hparams.train_logger.log_stats(
            stats_meta={"Epoch loaded for prior": self.hparams.epoch_counter.current},
        )
        dataloader = self.make_dataloader(train_data, **loader_kwargs, stage=sb.Stage.TEST)
        with torch.no_grad():
            num_classes = self.hparams.num_units
            total_occurences = torch.zeros((num_classes,))
            for batch in tqdm.tqdm(dataloader):
                occurences = torch.nn.functional.one_hot(batch.ali.data.long(), num_classes = num_classes)
                lengths = batch.ali.lengths*occurences.shape[1]
                mask = sb.dataio.dataio.length_to_mask(lengths)
                total_occurences += torch.sum(occurences * mask.unsqueeze(-1), dim=(0,1))
            if any(count == 0 for count in total_occurences):
                logger.warn("Zero count detected, using EPS")
                EPS = 1e-5
                log_prior = ((total_occurences+EPS) / total_occurences.sum()).log()
            else:
                log_prior = (total_occurences / total_occurences.sum()).log()
            return log_prior

            


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
            .rename(feats="feats.pth", ali="ali.pth")
            .repeat()
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                **hparams["dynamic_batch_kwargs"]
            )
    )
    validdata = (
            wds.WebDataset(hparams["validshards"])
            .decode()
            .rename(feats="feats.pth", ali="ali.pth")
            .batched(hparams["valid_batchsize"], collation_fn=sb.dataio.batch.PaddedBatch)
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
        opt_class=hparams["opt_class"],
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
