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
import matplotlib as mpl
import tensorflow as tf
import seaborn as sns

mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["axes.grid"] = False


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

    wv: pd.Series[float] = df.loc[:, "wv (m/s)"]
    bad_wv: pd.Series = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv: pd.Series[float] = df.loc[:, "max. wv (m/s)"]
    max_bad_wv: pd.Series = max_wv == -9999.0
    max_wv[max_bad_wv] = 0.0

    # print(df["wv (m/s)"].min())
    # print(df["max. wv (m/s)"].min())

    # create wind vector.
    wv = df.pop("wv (m/s)")
    max_wv = df.pop("max. wv (m/s)")

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
    https://pandas.pydata.org/docs/user_guide/visualization.html
    https://librarycarpentry.org/library-python/06a-plotting-with-pandas/
    https://www.atlassian.com/data/notebook/how-to-save-a-plot-to-a-file-using-matplotlib

    TODO

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
    plt.close()


def hello_world_txt() -> str:
    """
    read the hello_world.txt from input folder

    :return text content of the text file
    """
    with open(
        f"{os.getenv('TEST_SRCDIR')}/{os.getenv('TEST_WORKSPACE')}/src/time_series_forecasting/tests/input/hello_world.txt",
        "r",
    ) as f:
        return f.read()


def transpose() -> pd.DataFrame:
    """
    see at the statistics of the dataset

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
    return df.describe().transpose()


def wind_plot() -> None:
    """
    The last column of the data, wd (deg)—gives the wind direction in units of degrees.
    Angles do not make good model inputs: 360° and 0° should be close to each other and
    wrap around smoothly. Direction shouldn't matter if the wind is not blowing.

    Right now the distribution of wind data looks like this.

    But this will be easier for the model to interpret if you convert the wind direction and
    velocity columns to a wind vector.

    The distribution of wind vectors is much simpler for the model to correctly interpret.

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

    wv: pd.Series[float] = df.loc[:, "wv (m/s)"]
    bad_wv: pd.Series = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv: pd.Series[float] = df.loc[:, "max. wv (m/s)"]
    max_bad_wv: pd.Series = max_wv == -9999.0
    max_wv[max_bad_wv] = 0.0

    plt.hist2d(df["wd (deg)"], df["wv (m/s)"], bins=(50, 50), vmax=400)
    plt.colorbar()
    plt.xlabel("Wind Direction [deg]")
    plt.ylabel("Wind Velocity [m/s]")
    plt.savefig(fname=f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/wind_plot.png")
    plt.close()

    # create wind vector
    wv = df.pop("wv (m/s)")
    max_wv = df.pop("max. wv (m/s)")

    # convert to radians
    wd_rad: pd.Series = df.pop("wd (deg)") * np.pi / 180

    # Calculate the wind x and y components.
    df["Wx"] = wv * np.cos(wd_rad)
    df["Wy"] = wv * np.sin(wd_rad)
    # Calculate the max wind x and y components.
    df["max Wx"] = max_wv * np.cos(wd_rad)
    df["max Wy"] = max_wv * np.sin(wd_rad)

    plt.hist2d(df["Wx"], df["Wy"], bins=(50, 50), vmax=400)
    plt.colorbar()
    plt.xlabel("Wind X [m/s]")
    plt.ylabel("Wind Y [m/s]")
    ax = plt.gca()
    ax.axis("tight")
    plt.savefig(fname=f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/wind_vector_plot.png")
    plt.close()


def date_time_plot() -> None:
    """
    Similarly, the Date Time column is very useful, but not in this string form.
    Start by converting it to seconds.

    You can get usable signals by using sine and cosine transforms to clear
    "Time of day" and "Time of year" signals.

    If you don't have that information, you can determine which frequencies are important by
    extracting features with Fast Fourier Transform. To check the assumptions,
    here is the tf.signal.rfft of the temperature over time.

    Note the obvious peaks at frequencies near 1/year and 1/day.

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
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    year = (365.2425) * day

    df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
    df["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))

    plt.plot(np.array(df["Day sin"])[:25])
    plt.plot(np.array(df["Day cos"])[:25])

    plt.xlabel("Time [h]")
    plt.title("Time of day signal")
    plt.savefig(fname=f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/date_time_signal_plot.png")
    plt.close()

    fft: complex = tf.signal.rfft(df["T (degC)"])
    # print(type(fft))
    f_per_dataset: np.ndarray[int, np.dtype[np.int32]] = np.arange(0, len(fft))

    n_samples_h = len(df["T (degC)"])
    hours_per_year = 24 * 365.2524
    years_per_dataset = n_samples_h / (hours_per_year)
    f_per_year: np.ndarray[float, np.dtype[np.float32]] = f_per_dataset / years_per_dataset

    plt.step(f_per_year, np.abs(fft))
    plt.xscale("log")
    plt.ylim(0, 400000)
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 365.2524], labels=["1/year", "1/day"])
    _ = plt.xlabel("Frequency (log scale)")
    plt.savefig(fname=f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/date_time_rfft_plot.png")
    plt.close()


def normalize_plot() -> None:
    """
    Now, peek at the distribution of the features. Some features do have long tails, but there are
    no obvious errors like the -9999 wind velocity value.

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

    wv: pd.Series[float] = df.loc[:, "wv (m/s)"]
    bad_wv: pd.Series = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv: pd.Series[float] = df.loc[:, "max. wv (m/s)"]
    max_bad_wv: pd.Series = max_wv == -9999.0
    max_wv[max_bad_wv] = 0.0

    # print(df["wv (m/s)"].min())
    # print(df["max. wv (m/s)"].min())

    # create wind vector.
    wv: pd.Series[float] = df.pop("wv (m/s)")
    max_wv: pd.Series[float] = df.pop("max. wv (m/s)")

    # Convert to radians.
    wd_rad: pd.Series[float] = df.pop("wd (deg)") * np.pi / 180

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

    df_std: pd.DataFrame = (df - train_mean) / train_std
    df_std = df_std.melt(var_name="Column", value_name="Normalized")
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x="Column", y="Normalized", data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    plt.savefig(fname=f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/normalized_plot.png")
    plt.close()
