from pathlib import Path
from joblib import dump

import click
import numpy as np
import pandas as pd
import mlflow

# import mlflow.sklearn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold, GridSearchCV
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning


from .data import get_dataset
from .pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--m",
    default="logreg",
    type=str,
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--use-tsr",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--fs",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--s",
    default="ss",
    type=str,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=500,
    type=int,
    show_default=True,
)
@click.option(
    "--foldcnt",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--criterion",
    default="entropy",
    type=str,
    show_default=True,
)
@click.option(
    "--n",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=15,
    type=int,
    show_default=True,
)
@click.option(
    "--use-ncv",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--pca",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-f",
    default=0.5,
    show_default=True,
)
def kaggle(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    use_tsr: bool,
    max_iter: int,
    foldcnt: int,
    logreg_c: float,
    m: str,
    s: str,
    n: int,
    criterion: str,
    max_depth: int,
    fs: bool,
    use_ncv: bool,
    pca: bool,
    max_f: float
) -> None:
    features_train, target_train = get_dataset(
                    dataset_path, random_state, test_split_ratio, use_tsr
                )
    pipeline = create_pipeline(
            use_scaler,
            s,
            m,
            max_iter,
            logreg_c,
            random_state,
            n,
            criterion,
            max_depth,
            fs,
            # pca,
            max_f

        )
    
    mlflow.log_param("model", pipeline["classifier"])
    mlflow.log_param("N of folds", foldcnt)
    mlflow.log_param("use_scaler", use_scaler)
    mlflow.log_param("scaler", pipeline["scaler"])
    if fs:
        mlflow.log_param("feature_selection", pipeline["feature_selection"])
    if m == "logreg":
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
    elif m == "rf":
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n)
        mlflow.log_param("criterion", criterion)
    pipeline.fit(features_train, target_train)
    X = pd.read_csv(r"data\test.csv")
    preds = pipeline.predict(X)
    sample_submission  = pd.read_csv(r"data\sampleSubmission.csv")
    submission = pd.concat([sample_submission,pd.DataFrame(preds)], axis=1).drop(columns=['Cover_Type'])
    submission.columns = ['Id', 'Cover_Type']
    submission.to_csv('out.csv', index=False)
