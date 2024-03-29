import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader

from settings import Config, SplitType
from src.dataset import PosDataset, label2idx
from src.model import PosModel

logging.basicConfig(level=logging.INFO)
console_logger = logging.getLogger("fit_and_evaluate")


config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(config: Config):
    """
    Main function of the script. Parameters can be passed through modification of the default Config object
    using the Hydra library.
    """

    console_logger.info("Starting fit and evaluate script...")

    if not config.fit:
        console_logger.info("Not fitting model, loading with checkpoints...")
        # Instantiating trainer for evaluation
        trainer = pl.Trainer(**config.trainer)  # type: ignore
        if config.model.model_name is None:
            raise ValueError(
                f"Must specify a valid name of a model in directory {config.model.model_dir}"
            )
        else:
            final_model_dir = config.model.model_dir / config.model.model_name
            # There should be a single `.ckpt` file in the model directory, so take the first one that is found
            final_model_checkpoint = next(
                possible_checkpoint
                for possible_checkpoint in final_model_dir.iterdir()
                if possible_checkpoint.suffix == ".ckpt"
            )
        model = PosModel(config)

    else:
        console_logger.info("Started fitting of model...")

        # Load datasets and dataloaders
        train_dataset = PosDataset(config=config, split=SplitType.train)
        val_dataset = PosDataset(config=config, split=SplitType.dev)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.model.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.model.batch_size,
            num_workers=config.num_workers,
        )

        # Load model, callbacks, logger and trainer
        model = PosModel(config=config)
        if config.enable_wandb:
            logger = WandbLogger(
                project="pos-tagger",
                log_model="best",
                save_dir=str(config.wandb_dir),
            )
            experiment_name = logger.experiment.name
            logger.watch(model, log="all")
        else:
            experiment_name = None
        config.model.model_name = experiment_name if experiment_name else "local_run"
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.model.model_dir / config.model.model_name,
            monitor="val_loss",
            mode="min",
            save_weights_only=True,
            save_top_k=1,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=0,
        )
        trainer = pl.Trainer(  # type: ignore
            **config.trainer,
            callbacks=[checkpoint_callback, early_stopping],
            logger=logger if config.enable_wandb else False,
        )

        # Start fitting
        trainer.fit(
            model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )

        final_model_checkpoint = Path(checkpoint_callback.best_model_path)

    if not config.evaluate:
        console_logger.info("Not evaluating model, exiting...")
        return
    else:
        console_logger.info("Evaluating model...")
        test_dataset = PosDataset(config=config, split=SplitType.test)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.model.batch_size,
            num_workers=config.num_workers,
        )
        batched_predictions = trainer.predict(
            model,
            dataloaders=test_dataloader,
            return_predictions=True,
            ckpt_path=str(final_model_checkpoint),  # type: ignore
        )

        flattened_predictions = []
        flattened_targets = []
        for prediction_batch in batched_predictions:  # type: ignore
            for idx, tags in prediction_batch:
                # The tags returned by the model are already aligned with the input word separation, so we can directly
                # add them to the flattened list of predictions
                flattened_predictions += tags
                # We retrieve the targets from the index of the test dataset that is passed through the model. The dataset
                # already tokenizes the inputs and aligns the targets so we must recover the original targets, which are
                # the elements different from -100, before adding them to the flattened targets.
                full_targets = test_dataset[idx][-1]
                flattened_targets += full_targets[full_targets != -100].cpu().tolist()

        # Generate the classic scikit-learn classification report, which measures precision, recall and F1-score for all
        # target classes, together with the macro and weighted average.
        report = classification_report(
            flattened_targets,
            flattened_predictions,
            digits=4,
            labels=list(label2idx.values()),
            target_names=list(label2idx.keys()),
            zero_division=0,
        )
        # Generates the matplotlib plot of a confusion matrix useful to visualise the overall performance across classes.
        conf_matrix_plot = ConfusionMatrixDisplay.from_predictions(
            flattened_targets,
            flattened_predictions,
            labels=list(label2idx.values()),
            display_labels=list(label2idx.keys()),
            xticks_rotation="vertical",
            normalize="true",
        )
        # Build output directory and save classification report as well as confusion matrix
        output_dir = config.model.model_dir / config.model.model_name
        output_dir.mkdir(exist_ok=True, parents=True)
        with open(output_dir / "classification_report.txt", "w+") as f:
            f.write(report)
        conf_matrix_plot.figure_.savefig(output_dir / "confusion_matrix.png", dpi=700)

        # Save files to W&B
        if config.fit:
            wandb.save(str(final_model_checkpoint), policy="now")  # type: ignore
            wandb.save(str(output_dir / "classification_report.txt"), policy="now")  # type: ignore
            wandb.save(str(output_dir / "confusion_matrix.png"), policy="now")  # type: ignore


if __name__ == "__main__":
    main()
