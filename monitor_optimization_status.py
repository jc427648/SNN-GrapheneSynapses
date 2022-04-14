import logging
import sys
import optuna


# Check study status
study_names = ["10", "30", "100", "300", "500"]
for study_name in study_names:
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.load_study(
        study_name=None, storage=storage_name
    )
    print(study.direction)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)
    df.to_csv('%s.csv' % study_name)