from pathlib import Path

import joblib
import optuna
import torch
from torch.functional import Tensor
import torch.utils.data
from catboost.core import CatBoostClassifier
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from models import TTGAN

SEED = 123
CLASSIFIER_TYPE = "catboost"
OPTUNA_TRIALS = 50
OPTUNA_SEARCH_SPACE = {
    "epochs": {"low": 100, "high": 501, "step": 200},
    "translation_loss_weight": {"low": 0, "high": 0.19, "step": 0.05},
    "gen_pts_kept": {"low": 0.5, "high": 16, "log": True},
    "max_nn_prob": {"low": 0.6, "high": 1.05, "step": 0.2},
    "identity_loss_weight": {"low": 0, "high": 11, "step": 2.5},
    "cyclic_loss_weight": {"low": 0, "high": 16, "step": 5},
}
BATCH_SIZE = 4096

TRAIN_DATA = Path(r"train.pt")  # tuple of X, y tensors
VAL_DATA = Path(r"val.pt")
TEST_DATA = Path(r"test.pt")
RESULTS_DIR = Path("results")


def create_classifier(type):
    if type == "catboost":
        return CatBoostClassifier(
            custom_loss=["F1"],
            random_seed=123,
            learning_rate=0.2,
            iterations=100,
            depth=6,
        )
    if type == "svm":
        return SVC(kernel="linear", probability=True)


def train_and_evaluate(
    train_data: Tensor,
    test_data: Tensor,
    gan_model_path: Path,
    trial: bool,
    **hparams
):
    x_train, y_train = train_data
    x_test, y_test = test_data
    x_maj, x_min = x_train[y_train == 0], x_train[y_train == 1]
    translation = (
        hparams["cyclic_loss_weight"]
        or hparams["translation_loss_weight"]
        or hparams["identity_loss_weight"]
    )
    if not gan_model_path.exists():
        ds_min = TensorDataset(x_min.float())
        if translation:
            ds_maj = TensorDataset(x_maj.float())
        model = TTGAN(
            latent_dim=x_train.shape[1],
            output_dim=x_train.shape[1],
            lr=1e-4,
            translation_loss_weight=hparams["translation_loss_weight"],
            cyclic_loss_weight=hparams["cyclic_loss_weight"],
            identity_loss_weight=hparams["identity_loss_weight"],
        )
        trainer = Trainer(
            gpus=-1,
            max_epochs=hparams["epochs"],
            multiple_trainloader_mode="min_size",
        )
        batch_size = min(BATCH_SIZE, x_min.shape[0])
        train_dataloaders = {
            "min": DataLoader(ds_min, batch_size=batch_size, shuffle=True)
        }
        if translation:
            train_dataloaders.update(
                {
                    "maj": DataLoader(
                        ds_maj, batch_size=batch_size, shuffle=True
                    )
                }
            )
        trainer.fit(model, train_dataloaders=train_dataloaders)
        trainer.save_checkpoint(gan_model_path)
    model = TTGAN.load_from_checkpoint(gan_model_path, strict=False).cuda()
    if not translation:
        x_gen = (
            model(torch.randn(x_maj.shape[0], x_maj.shape[1]).cuda())
            .cpu()
            .detach()
        )
    else:
        x_gen = model(x_maj.float().cuda()).cpu().detach()
    classifier_points_filter = create_classifier(type=CLASSIFIER_TYPE)
    classifier_points_filter.fit(x_train.numpy(), y_train.numpy())
    probas = classifier_points_filter.predict_proba(x_gen.numpy())[:, 1]
    x_gen = x_gen[(probas >= 0.05) & (probas <= hparams["max_nn_prob"])]
    if x_gen.shape[0] == 0:
        raise optuna.TrialPruned()
    x_gen = x_gen[
        classifier_points_filter.predict_proba(x_gen.numpy())[:, 1]
        .argsort()[::-1]
        .copy()
    ]
    x_gen = x_gen[: int(hparams["gen_pts_kept"] * x_min.shape[0])]
    if x_gen.shape[0] == 0:
        raise optuna.TrialPruned()

    x_all = torch.cat([x_train] + [x_gen])
    y_all = torch.cat([y_train] + [torch.ones(len(x_gen))])
    x_all, y_all = x_all.float().numpy(), y_all.float().numpy()
    x_test, y_test = x_test.numpy(), y_test.numpy()
    classifier = create_classifier(type=CLASSIFIER_TYPE)
    classifier.fit(x_all, y_all)
    if trial:
        return average_precision_score(
            y_test, classifier.predict_proba(x_test)[:, 1]
        )
    else:
        return (
            x_gen.shape[0],
            average_precision_score(
                y_test, classifier.predict_proba(x_test)[:, 1]
            ),
        )


def objective(trial: optuna.Trial, train_data: Path, val_data: Path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hparams = {
        name: trial.suggest_float(name, **values)
        if name != "epochs"
        else trial.suggest_int(name, **values)
        for name, values in OPTUNA_SEARCH_SPACE.items()
    }

    # we store GAN models as a sort of cache,
    # since some trials share the same GAN
    gan_model_path = Path(
        RESULTS_DIR
        / "gan_{}_epochs_{}_translation_{}_cyclic_{}_identity.ckpt".format(
            hparams["epochs"],
            hparams["translation_loss_weight"],
            hparams["cyclic_loss_weight"],
            hparams["identity_loss_weight"],
        )
    )
    return train_and_evaluate(
        train_data, val_data, gan_model_path, True, **hparams
    )


def run_study(study_save_path: Path, train_data, val_data):
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(
        lambda trial: objective(
            trial, train_data=train_data, val_data=val_data
        ),
        n_trials=OPTUNA_TRIALS,
    )
    joblib.dump(study, study_save_path)


if __name__ == "__main__":
    seed_everything(SEED, workers=True)
    x_train, y_train = torch.load(TRAIN_DATA)
    x_val, y_val = torch.load(VAL_DATA)
    x_train_total, y_train_total = (
        torch.cat((x_train, x_val)),
        torch.cat((y_train, y_val)),
    )
    x_test, y_test = torch.load(TEST_DATA)
    run_study(RESULTS_DIR / "study.pkl", (x_train, y_train), (x_val, y_val))
    study = joblib.load(RESULTS_DIR / "study.pkl")
    best_params = study.best_params
    gan_model_path = Path(RESULTS_DIR / "final_gan_optuna_best.ckpt")
    num_generated_points, ap = train_and_evaluate(
        (x_train_total, y_train_total),
        (x_test, y_test),
        gan_model_path,
        False,
        **best_params
    )
    Path(RESULTS_DIR / "optuna_ap.txt").write_text(str(ap))
    Path(RESULTS_DIR / "optuna_best_params.txt").write_text(
        "{} number of synth points: {}".format(
            best_params, num_generated_points
        )
    )
