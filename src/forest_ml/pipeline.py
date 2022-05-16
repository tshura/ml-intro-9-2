from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

def create_pipeline(
    use_scaler: bool,
    s: str,
    m: str,
    max_iter: int,
    logreg_C: float,
    random_state: int,
    n: int,
    crit: str,
    max_depth: int,
    fs: bool,
    # pca: bool
    max_f: float
) -> Pipeline:
    pipeline_steps = []
    if m == "logreg":
        model = LogisticRegression(
            random_state=random_state, max_iter=max_iter, C=logreg_C
        )
    elif m == "rf":
        model = RandomForestClassifier(
            random_state=random_state,
            max_depth=max_depth,
            n_estimators=n,
            criterion=crit,
            max_features = max_f
        )
    if use_scaler:
        if s == "ss":
            pipeline_steps.append(("scaler", StandardScaler()))
        elif s == "mm":
            pipeline_steps.append(("scaler", MinMaxScaler()))
    if fs:
        pipeline_steps.append(
            (
                "feature_selection",
                SelectFromModel(
                    LogisticRegression(C=1, max_iter=max_iter, penalty="l2")
                    # RandomForestClassifier(random_state=random_state,
            # max_depth=max_depth,
            # n_estimators=n,
            # criterion=crit,
            # max_features = max_f)
                ),
            )
        )
    # if pca:
    #     pipeline_steps.append(
    #         (
    #             "feature_selection",
    #             PCA(n_components = 0.95)
    #         )
    #     )
    pipeline_steps.append(("classifier", model))

    return Pipeline(steps=pipeline_steps)

