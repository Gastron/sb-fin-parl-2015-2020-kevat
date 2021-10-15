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

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class XentAM(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        normalized = self.modules.normalize(batch.feats, lengths=torch.ones(batch.feats.shape[0]))
        encoded_all = self.modules.encoder(normalized)
        # Batches have some context in them (to help), 
        # but this should not be predicted:
        encoded_relevant = encoded_all[:,self.hparams.front_index:self.hparams.back_index,:]
        out = self.modules.lin_out(encoded_relevant)
        predictions = self.hparams.log_softmax(out)
        return predictions

    def compute_objectives(self, predictions, batch, stage):
        loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions,
            targets=batch["ali"],
            label_smoothing=self.hparams.label_smoothing,
        )
        if stage != sb.Stage.TRAIN:
            self.accuracy_metric.append(predictions, batch["ali"])
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
                meta={"accuracy": stage_stats["accuracy"]}, min_keys=["accuracy"],
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

    def estimate_prior(self, train_data, loader_kwargs={}, max_key=None, min_key=None):
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        dataloader = self.make_dataloader(train_data, **loader_kwargs, stage=sb.Stage.TEST)
        with torch.no_grad():
            prior = torch.zeros((self.hparams.num_units,))
            num_predictions = 0
            for batch in tqdm.tqdm(dataloader):
                predictions = self.compute_forward(batch, stage=sb.Stage.TEST) 
                summed_preds = torch.sum(predictions, dim=(0,1))
                prior += summed_preds.detach().cpu()
                num_predictions += predictions.shape[0]*predictions.shape[1]
            prior = prior / num_predictions
        return prior
            

            


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
            .batched(hparams["train_batchsize"], collation_fn=local.Batch)
    )
    validdata = (
            wds.WebDataset(hparams["validshards"])
            .decode()
            .rename(feats="feats.pth", ali="ali.pth")
            .batched(hparams["valid_batchsize"], collation_fn=local.Batch)
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
        ckpt = hparams["ckpt_finder"].find_checkpoint(min_key="WER")
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
    prior = asr_brain.estimate_prior(
            datasets["train"], 
            loader_kwargs=hparams["prior_loader_kwargs"],
            max_key=hparams["test_max_key"]
    )
    torch.save(prior, hparams["prior_file"])
