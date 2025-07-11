import polars as pl

base_path = "/kaggle/input/mtcaic3"

train_df = pl.read_csv("/kaggle/input/mtcaic3/train.csv")
validation_df = pl.read_csv("/kaggle/input/mtcaic3/validation.csv")
test_df = pl.read_csv("/kaggle/input/mtcaic3/test.csv")


def get_trials(data):
    return [data.row(i, named=True) for i in range(data.height)]


def load_data(trial):
    id_num = trial["id"]

    if id_num <= 4800:
        type_dataset = "train"
    elif id_num <= 4900:
        type_dataset = "validation"
    else:
        type_dataset = "test"

    task = trial["task"]
    subject_id = trial["subject_id"]
    trial_session = trial["trial_session"]
    trial_num = trial["trial"]
    label = trial["label"]

    path = f"{base_path}/{task}/{type_dataset}/{subject_id}/{trial_session}/EEGdata.csv"

    data = pl.read_csv(path)

    s_r = 2250 if task == "MI" else 1750

    s_i = (trial_num - 1) * s_r

    trial_data = data.slice(s_i, s_r)

    return trial_data


def get_data(trials):
    return [load_data(trial) for trial in trials]


train_trials = get_trials(train_df)
test_trials = get_trials(test_df)
validation_trials = get_trials(validation_df)

new_data_trained = get_data(train_trials)

new_data_test = get_data(test_trials)

new_data_validation = get_data(validation_trials)
