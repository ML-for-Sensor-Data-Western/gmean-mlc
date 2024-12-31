"""
Tuning selected parameters using Ray Tune.

The parameters in GLOBAL_CONFIG are the total hyperparameters that can be tuned.
The same parameters are in terminal args as well.
Make sure the parameters which do not need to be tuned are properly defined in
terminal args and the parameters to be tuned (must be a subset of GLOBAL_CONFIG)
are stated in args.params.

Refer https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html#pytorch-lightning-classifier-for-mnist
for onboarding ray train if each trial is a distributed training job.
"""

import os
from argparse import ArgumentParser

import lightning.pytorch as pl
import torch
from ray import tune
from ray.train import CheckpointConfig, RunConfig
from ray.tune import CLIReporter, Tuner
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torchvision import transforms

from lightning_datamodules import MultiLabelDataModule
from lightning_model import MultiLabelModel
from loss import HybridLoss

torch.set_float32_matmul_precision("high")

GLOBAL_CONFIG = {
    "batch_size": tune.choice([64, 128, 256]),
    "learning_rate": tune.choice([0.001, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]),
    "lr_decay": tune.choice([0.01, 0.05, 0.1]),
    "momentum": tune.choice([0.5, 0.6, 0.7, 0.8, 0.9]),
    "weight_decay": tune.uniform(0.0001, 0.01),
    "class_balancing_beta": tune.choice([0.9, 0.99, 0.999, 0.9999]),
    "meta_loss_weight": tune.uniform(0.0, 1.0),
    "meta_loss_beta": tune.uniform(0.0, 1.0),
}


def train(config, args):
    # pl.seed_everything(1234567890)
    args_dict = vars(args)

    hyperparameters = {}
    for hp_name in GLOBAL_CONFIG.keys():
        if hp_name in config.keys():
            hyperparameters[hp_name] = config[hp_name]
        elif hp_name in args_dict.keys():
            hyperparameters[hp_name] = args_dict[hp_name]
        else:
            raise Exception(f"Hyperparam underfined in args '{hp_name}'")

    img_size = 299 if args.model in ["inception_v3", "chen2018_multilabel"] else 224

    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154]),
        ]
    )

    dm = MultiLabelDataModule(
        batch_size=hyperparameters["batch_size"],
        workers=args.workers,
        ann_root=args.ann_root,
        data_root=args.data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        only_defects=False,
    )

    dm.prepare_data()
    dm.setup("fit")

    criterion = HybridLoss(
        class_counts=dm.class_counts,
        normal_count=dm.num_train_samples - dm.defect_count,
        class_balancing_beta=hyperparameters["class_balancing_beta"],
        base_loss=args.base_loss,
        focal_gamma=args.focal_gamma,
        meta_loss_weight=hyperparameters["meta_loss_weight"],
        meta_loss_beta=hyperparameters["meta_loss_beta"],
    )

    light_model = MultiLabelModel(
        model=args.model,
        num_classes=dm.num_classes,
        criterion=criterion,
        learning_rate=hyperparameters["learning_rate"],
        momentum=hyperparameters["momentum"],
        weight_decay=hyperparameters["weight_decay"],
        batch_size=hyperparameters["batch_size"],
        lr_steps=args.lr_steps,
    )
    

    tune_callback = TuneReportCheckpointCallback(
        metrics={args.metric: args.metric}, on="validation_end"
    )

    trainer = pl.Trainer(
        precision=args.precision,
        max_epochs=args.max_epochs,
        benchmark=True,
        callbacks=[tune_callback],
        enable_progress_bar=False,
    )

    trainer.fit(light_model, dm)


def trial_name(trial):
    return str(trial.trial_id)


def trial_dirname(trial):
    return "trial_" + str(trial.trial_id)


def tune_parameters(args):
    print("\nAll args:", args)

    # Adjust learning rate to amount of GPUs
    # args.workers = max(0, min(8, 4 * len(args.gpus_per_trial)))
    # args.learning_rate = args.learning_rate * (len(args.gpus_per_trial) * args.batch_size) / 256

    config = {}
    for param in args.params:
        if param in GLOBAL_CONFIG.keys():
            config[param] = GLOBAL_CONFIG[param]
        else:
            raise Exception(f"Hyperparam undefined in global config '{param}'")

    print("\ntunable parameters: ", config.keys())

    metric_optim_mode = (
        "max" if args.metric in ["val_ap", "val_f1", "val_f2"] else "min"
    )

    search_alg = OptunaSearch(metric=args.metric, mode=metric_optim_mode)

    search_scheduler = ASHAScheduler(
        time_attr="training_iteration",  # default
        max_t=100,  # default
        grace_period=18,
        reduction_factor=4,  # default
        brackets=1,  # default
        stop_last_trials=True,  # default
    )

    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute=args.metric,
        checkpoint_score_order=metric_optim_mode,
    )

    tune_config = tune.TuneConfig(
        metric=args.metric,
        mode=metric_optim_mode,
        search_alg=search_alg,
        scheduler=search_scheduler,
        num_samples=args.num_trials,
        max_concurrent_trials=args.max_concurrent_trials,
        trial_name_creator=trial_name,
        trial_dirname_creator=trial_dirname,
    )

    run_config = RunConfig(
        name="e2e-version_%s" % (args.log_version),
        storage_path=os.path.join(args.log_save_dir, args.model),
        checkpoint_config=checkpoint_config,
        verbose=2,
    )

    trainable = tune.with_resources(
        trainable=tune.with_parameters(train, args=args),
        resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
    )

    tuner = Tuner(
        trainable=trainable,
        param_space=config,
        tune_config=tune_config,
        run_config=run_config,
    )

    result = tuner.fit()

    print(
        "The best param values are: ",
        result.get_best_result(metric=args.metric, mode=metric_optim_mode),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ann_root", type=str, default="./annotations")
    parser.add_argument("--data_root", type=str, default="./Data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--max_concurrent_trials", type=int, default=1)
    parser.add_argument("--cpus_per_trial", type=int, default=8)
    parser.add_argument("--gpus_per_trial", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_steps", nargs="+", type=int, default=[20, 30])
    parser.add_argument("--lr_decay", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=MultiLabelModel.MODEL_NAMES
    )
    parser.add_argument("--mtl_heads", action="store_true")
    parser.add_argument(
        "--base_loss", type=str, default="focal", choices=["focal", "bce"]
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--class_balancing_beta", type=float, default=0.9999)
    parser.add_argument("--meta_loss_weight", type=float, default=0.1)
    parser.add_argument("--meta_loss_beta", type=float, default=0.1)
    parser.add_argument(
        "--metric",
        type=str,
        help="metric to optimizer",
        choices=["val_ap", "val_f1", "val_f2"],
    )
    parser.add_argument(
        "--params",
        nargs="+",
        type=str,
        help="params to optimize",
        default=["meta_loss_weight", "meta_loss_beta", "class_balancing_beta"],
    )
    parser.add_argument("--log_save_dir", type=str, default="./logs")
    parser.add_argument("--log_version", type=int, default=1)

    args = parser.parse_args()

    tune_parameters(args)
