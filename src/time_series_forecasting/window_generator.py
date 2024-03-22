#
#  window_generator.py
#
#  Copyright 2024 Pepe Unlimited
#  Licensed under the MIT license, see associated LICENSE file for terms.
#  See AUTHORS file for the list of project authors.
#

from typing import List
import requests
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import os


class WindowGenerator:
    hello_world: str = "Hello, World!"

    # inputs
    input_width: int = 0
    label_width: int = 0
    shift: int = 0

    # data
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

    # columns
    label_columns: List[str]
    label_columns_indices: dict[str, int]
    columns_indices: dict[pd.Index, int]

    # parameters
    total_window_size: int
    input_slice: slice
    input_slices: np.ndarray[int, np.dtype[np.int64]]
    label_start: int
    label_slices: slice
    label_indices: np.ndarray[int, np.dtype[np.int64]]

    def __init__(
        self,
        input_width: int = 0,
        label_width: int = 0,
        shift: int = 0,
        train_df: pd.DataFrame = pd.DataFrame(),
        val_df: pd.DataFrame = pd.DataFrame(),
        test_df: pd.DataFrame = pd.DataFrame(),
        label_columns: List[str] = [],
    ):
        """
        https://www.tensorflow.org/api_docs/python/tf/data/Dataset
        https://pandas.pydata.org/pandas-docs/stable/reference/frame.html
        https://numpy.org/doc/stable/reference/arrays.ndarray.html

        :param input_width: width (number of time steps) of the input and label windows
        :param label_width: ???
        :param shift: time offset between them.

        :param train_df: time offset between them.
        :param val_df: time offset between them.
        :param test_df: time offset between them.

        :param label_columns: which features are used as inputs, labels, or both
        """

        # data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # work out the label column indices.
        self.label_columns = label_columns
        if not label_columns:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_slices = np.arange(self.total_window_size, dtype=np.int64)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.label_slices = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size, dtype=np.int64)[self.label_slices]

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_slices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def printer(self):
        print(sys.version)
        print(requests.__version__)
        print(tf.__version__)
        print(pd.__version__)
        print(np.__version__)
        print(keras.__version__)

    def weather_dataset(self):
        zip_path = keras.utils.get_file(
            origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
            fname="jena_climate_2009_2016.zip",
            extract=True,
        )

        csv_path, _ = os.path.splitext(f"{zip_path}.csv")
        df = pd.read_csv(csv_path)
        # Slice [start:stop:step], starting from index 5 take every 6th record.
        df = df[5::6]

        wv = df["wv (m/s)"]
        bad_wv = wv == -9999.0
        wv[bad_wv] = 0.0

        # print(df["wv (m/s)"].min())

        wv = df.pop("wv (m/s)")
        max_wv = df.pop("max. wv (m/s)")

        # Convert to radians.
        wd_rad = df.pop("wd (deg)") * np.pi / 180

        # Calculate the wind x and y components.
        df["Wx"] = wv * np.cos(wd_rad)
        df["Wy"] = wv * np.sin(wd_rad)
        # Calculate the max wind x and y components.
        df["max Wx"] = max_wv * np.cos(wd_rad)
        df["max Wx"] = max_wv * np.sin(wd_rad)

        # convert to seconds
        date_time = pd.to_datetime(df.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")
        timestamp_s = date_time.map(pd.Timestamp.timestamp)
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
        train_df = df[0 : int(n * 0.7)]
        val_df = df[int(n * 0.7) : int(n * 0.9)]
        test_df = df[int(n * 0.9) :]

        # num_features = df.shape[1]
        # print(num_features)

        # normalize the data
        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std

        # for col in df.columns:
        #    print(col)
        # print(df.head())
