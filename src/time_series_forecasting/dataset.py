#
#  dataset.py
#
#  Copyright 2024 Pepe Unlimited
#  Licensed under the MIT license, see associated LICENSE file for terms.
#  See AUTHORS file for the list of project authors.
#

import pandas as pd
import numpy as np
import keras
import os
import matplotlib.pyplot as plt


def weather_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    asdasd

    :return
    """
    zip_path = keras.utils.get_file(
        origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        fname="jena_climate_2009_2016.zip",
        extract=True,
    )

    csv_path, _ = os.path.splitext(f"{zip_path}.csv")
    df: pd.DataFrame = pd.read_csv(csv_path)
    # Slice [start:stop:step], starting from index 5 take every 6th record.
    df = df.loc[5::6]

    wv: pd.Series = df.loc[:, "wv (m/s)"]
    bad_wv: pd.Series = wv == -9999.0
    wv[bad_wv] = 0.0

    # print(df["wv (m/s)"].min())

    wv: pd.Series = df.pop("wv (m/s)")
    max_wv: pd.Series = df.pop("max. wv (m/s)")

    # Convert to radians.
    wd_rad: pd.Series = df.pop("wd (deg)") * np.pi / 180

    # Calculate the wind x and y components.
    df["Wx"] = wv * np.cos(wd_rad)
    df["Wy"] = wv * np.sin(wd_rad)
    # Calculate the max wind x and y components.
    df["max Wx"] = max_wv * np.cos(wd_rad)
    df["max Wy"] = max_wv * np.sin(wd_rad)

    # convert to seconds
    date_time: pd.Series = pd.to_datetime(df.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")
    timestamp_s: pd.Series = date_time.map(pd.Timestamp.timestamp)
    # print(timestamp_s)

    # Time of day" and "Time of year" signals
    day = 24 * 60 * 60
    year = (365.2425) * day

    df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
    df["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))

    # split the data
    # column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df: pd.DataFrame = df.loc[0 : int(n * 0.7)]
    val_df: pd.DataFrame = df.loc[int(n * 0.7) : int(n * 0.9)]
    test_df: pd.DataFrame = df.loc[int(n * 0.9) :]

    # num_features = df.shape[1]
    # print(num_features)

    # normalize the data
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df: pd.DataFrame = (train_df - train_mean) / train_std
    val_df: pd.DataFrame = (val_df - train_mean) / train_std
    test_df: pd.DataFrame = (test_df - train_mean) / train_std

    # for col in df.columns:
    #    print(col)
    # print(df.head())

    return train_df, val_df, test_df


def hello_world_plot() -> None:
    """
    https://matplotlib.org/stable/api/index.html
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.index.html
    https://bazel.build/reference/test-encyclopedia

    TEST_UNDECLARED_OUTPUTS_DIR

    :return none
    """

    zip_path = keras.utils.get_file(
        origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        fname="jena_climate_2009_2016.zip",
        extract=True,
    )
    csv_path, _ = os.path.splitext(f"{zip_path}.csv")
    df: pd.DataFrame = pd.read_csv(csv_path)
    # Slice [start:stop:step], starting from index 5 take every 6th record.
    df = df.loc[5::6]

    date_time: pd.Series = pd.to_datetime(df.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")
    plot_cols = ["T (degC)", "p (mbar)", "rho (g/m**3)"]

    # plot_features: pd.DataFrame = df.loc[:, plot_cols]
    # plot_features.index = date_time

    plot_features = df.loc[:, plot_cols][:480]
    plot_features.index = date_time[:480]

    _ = plot_features.plot(subplots=True)

    plt.tight_layout()
    plt.savefig(fname=f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/hello_world_plot.png")


def hello_world_txt() -> str:
    with open(
        f"{os.getenv('TEST_SRCDIR')}/{os.getenv('TEST_WORKSPACE')}/src/time_series_forecasting/tests/input/hello_world.txt",
        "r",
    ) as f:
        return f.read()
