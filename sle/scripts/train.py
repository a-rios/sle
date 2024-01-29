import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor,TQDMProgressBar

from sle.data import SLEDataModule
from sle.model import RobertaFinetuner


if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()

    # add model specific args to parser
    parser = RobertaFinetuner.add_model_specific_args(parser)
    # args = parser.parse_args()

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # prepare data module and finetuner class
    if args.checkpoint is None:
        model = RobertaFinetuner(params=args)
    else:
        model = RobertaFinetuner.load_from_checkpoint(args.checkpoint, params=args, strict=False)

    dm = SLEDataModule(model.tokenizer, params=args)

    checkpoint_callback = None
    mode = "max" if args.ckpt_metric.split("_")[-1] == "f1" else "min"
    os.makedirs(os.path.dirname(os.path.join(args.save_dir, args.save_prefix)), exist_ok=True)
    if args.no_log:
        logger = False
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.save_dir, args.save_prefix))
    else:
        if args.name is None:
            # use default logger settings (for hparam sweeps)
            wandb_logger = WandbLogger()
        else:
            # prepare logger
            wandb_logger = WandbLogger(
                name=args.name, project=args.project, save_dir=args.save_dir, id=args.wandb_id)

            # checkpoint callback
            # checkpoint_callback = [ModelCheckpoint(monitor=args.ckpt_metric, mode=mode, save_last=True)]
            checkpoint_name = "{{epoch:02d}}_{{step:02d}}_{{{}".format(args.early_stopping_metric)
            checkpoint_name += ':.3f}'
            checkpoint_callback = ModelCheckpoint(
                    dirpath=os.path.join(args.save_dir, args.save_prefix),
                    filename=checkpoint_name,
                    save_top_k=args.save_top_k,
                    verbose=True,
                    monitor=args.early_stopping_metric,
                    mode=mode)
        logger = wandb_logger

    # add early stopping
    early_stop_callback = EarlyStopping(monitor=args.early_stopping_metric, min_delta=args.min_delta, patience=args.early_stopping_patience, verbose=True, mode=mode) # vloss or doc_mae
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    progress_bar_callback = TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate)

    trainer = pl.Trainer(
        val_check_interval=args.val_check_interval,
        logger=logger,
        accelerator="gpu",
        strategy="ddp",
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar_callback, lr_monitor_callback],
        precision=32 if args.fp32 else "16-mixed",
        enable_progress_bar=True)

    trainer.fit(model, dm)
    model.save_pretrained(args.save_dir + "/" + args.save_prefix)
    model.tokenizer.save_pretrained(args.save_dir + "/" + args.save_prefix)
