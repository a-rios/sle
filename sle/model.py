import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from scipy.stats import entropy
from sklearn.metrics import mean_absolute_error
# from transformers import AdamW, RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification


class RobertaFinetuner(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        # saves params to the checkpoint and in self.hparams
        self.save_hyperparameters(params)

        num_labels = 1 # regression
        self.hparams["num_labels"] = num_labels

        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams['model_name_or_path'], num_labels=num_labels, cache_dir=self.hparams['cache_dir'])
        print(f"Initial AutoModelForSequenceClassification model loaded from {self.hparams['model_name_or_path']}.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams['model_name_or_path'], cache_dir=self.hparams['cache_dir'])

        # training loss cache to log mean every n steps
        self.train_losses = []

        if "hidden_dropout_prob" in self.hparams and self.hparams.hidden_dropout_prob is not None:
            self.model.config.hidden_dropout_prob = self.hparams.hidden_dropout_prob # default in xmlroberta == 0.1
        self.validation_step_outputs = []

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self.model(**{k:v for k, v in batch.items() if k not in ["doc_ids"]}, return_dict=True)
        loss = output["loss"]
        self.train_losses.append(loss)

        # logging mean loss every `n` steps
        if batch_idx % int(self.hparams.train_check_interval * self.trainer.num_training_batches) == 0:
            avg_loss = torch.stack(self.train_losses).mean()
            self.log("train_avg_loss", avg_loss)
            self.train_losses = []
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self.model(**{k:v for k, v in batch.items() if k not in ["doc_ids"]}, return_dict=True)
        loss = output["loss"]
        logits = output["logits"].cpu()

        output = {
            "loss": loss,
            "preds": logits,
        }

        output["labels"] = batch["labels"]
        if self.has_param("log_doc_mae"):
            output["doc_ids"] = batch["doc_ids"]

        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self, prefix="val"):
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log(f"{prefix}_loss", loss)

        # compute document-level MAE when doing regression
        if self.has_param("log_doc_mae"):
            flat_preds = [y.item() for ys in self.validation_step_outputs for y in ys["preds"]]
            flat_labels = [y.item() for ys in self.validation_step_outputs for y in ys["labels"]]

            doc_ids = [y for ys in self.validation_step_outputs for y in ys["doc_ids"]]
            doc_preds = {}
            doc_labs = {}
            for i in range(len(doc_ids)):
                if doc_ids[i] not in doc_preds:
                    doc_preds[doc_ids[i]] = [flat_preds[i]]
                    doc_labs[doc_ids[i]] = [flat_labels[i]]
                else:
                    doc_preds[doc_ids[i]].append(flat_preds[i])
                    doc_labs[doc_ids[i]].append(flat_labels[i])
            doc_means = []
            doc_gts = []
            for k, v in doc_preds.items():
                doc_means.append(np.mean(v))
                doc_gts.append(np.mean(doc_labs[k]))
            self.log(f"{prefix}_doc_mae", mean_absolute_error(doc_gts, doc_means), on_step=False, on_epoch=True, prog_bar=False)
            self.log('vloss', loss, on_step=False, on_epoch=True, prog_bar=False)
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.01
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        mode = "max" if self.hparams['ckpt_metric'].split("_")[-1] == "f1" else "min"

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode=mode,
                factor=self.hparams['lr_reduce_factor'],
                patience=self.hparams['lr_reduce_patience'],
                cooldown=self.hparams['lr_cooldown'],
                verbose=True)
        self.lr_scheduler_config = {
                "scheduler": self.scheduler,
                "monitor": self.hparams['early_stopping_metric'],
                "interval": "step",
                "frequency": self.hparams['val_check_interval'],
                "strict": True,
                "name": "plateauLR"
        }
        return [self.optimizer], [self.lr_scheduler_config]

    def has_param(self, param):
        """Check if param exists and has a non-negative/null value."""
        if param in self.hparams:
            param = self.hparams[param] # set `param` to actual value
            if param is not None:
                if not isinstance(param, bool) or param:
                    return True
        return False

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"{type(self.model)} model saved to {path}.")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument("--name", type=str, default=None, required=False,)
        parser.add_argument("--project", type=str, default=None, required=False,)
        parser.add_argument("--save_dir", type=str, default=None, required=False,)
        parser.add_argument("--checkpoint", type=str, default=None, required=False,)
        parser.add_argument("--wandb_id", type=str, default=None, required=False,)

        parser.add_argument("--train_file", type=str, default=None, required=False)
        parser.add_argument("--val_file", type=str, default=None, required=False)
        parser.add_argument("--x_col", type=str, default="x", required=False,)
        parser.add_argument("--y_col", type=str, default="y", required=False,)
        parser.add_argument("--train_check_interval", type=float, default=0.20)
        parser.add_argument("--train_split", type=float, default=0.9)
        parser.add_argument("--val_split", type=float, default=0.05)

        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--max_samples", type=int, default=-1)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--train_workers", type=int, default=8)
        parser.add_argument("--val_workers", type=int, default=8)
        parser.add_argument("--max_length", type=int, default=128)
        parser.add_argument("--ckpt_metric", type=str, default="val_loss", required=False,)

        parser.add_argument("--hidden_dropout_prob", type=float, default=None, required=False,)

        parser.add_argument("--log_doc_mae", action="store_true")
        parser.add_argument("--no_log", action="store_true")

        # additional params
        parser.add_argument("--model_name_or_path", type=str, help="Pretrained model to load as AutoModelForSequenceClassification")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--grad_accum", type=int, default=1, help="Number of gradient accumulation steps.")
        parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs (will stop training even if patience for early stopping has not been reached). Default: 100.")
        parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of steps (will stop training even if patience for early stopping has not been reached). Default: -1 (no maximum).")
        parser.add_argument("--save_top_k", type=int, default=1, help="Number of best checkpoints to keep. Others will be removed.")
        parser.add_argument("--progress_bar_refresh_rate", type=int, default=0, help="How often to refresh progress bar (in steps). Value 0 disables progress bar.")
        parser.add_argument("--save_prefix", type=str, default='test', help="subfolder in save_dir for this model")


        # lr scheduler
        parser.add_argument("--lr_reduce_patience", type=int, default=8, help="Patience for LR reduction in Plateau scheduler. NOTE: if interval=steps, and lr_scheduler=ReduceLROnPlateau, frequency MUST be smaller than the number of batches per epoch, otherwise lr_scheduler.step() never gets called and lr is not reduced (because lightning calls step() in this case based on batch index, which is reset after each epoch).")
        parser.add_argument("--lr_reduce_factor", type=float, default=0.5, help="Learning rate reduce factor for Plateau scheduler.")
        parser.add_argument("--lr_cooldown", type=int, default=0, help="Cooldown for Plateau scheduler (number of epochs to wait before resuming normal operation after lr has been reduced.).")


        parser.add_argument("--cache_dir", type=str, metavar="PATH", help="Cache directory for huggingface models")
        parser.add_argument("--val_check_interval", type=int, help="How often to check the validation set in number of updates.")
        # early stopping
        parser.add_argument("--early_stopping_metric", type=str, default="vloss", help="Metric to monitor for early stopping. doc_mae or vloss")
        parser.add_argument("--min_delta", type=float, default=0, help="Minimum delta to be considered an improvement for early stopping")
        parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping")

        return parser
