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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as max


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

    def __repr__(self) -> str:
        """
        print(repr(object))
        """
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_slices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(self, features: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        https://www.geeksforgeeks.org/different-ways-to-create-pandas-dataframe/

        Features (xs)
        A feature is an **input** variable—the x variable in simple linear regression.
        A simple machine learning project might use a single feature, while a more sophisticated
        machine learning project could use millions of features, specified as:

        In the spam detector example, the features could include the following:
        - words in the email text
        - sender's address
        - time of day the email was sent
        - email contains the phrase "one weird trick."

        Labels (ys)
        A label is the thing we're **predicting—the** y variable in simple linear regression.
        The label could be:
        - the future price of wheat
        - the kind of animal shown in a picture,
        - the meaning of an audio clip, or just about anything.

        Typically, data in TensorFlow is packed into arrays where:
        - outermost index is across examples (the "batch" dimension)
        - middle indices are the "time" or "space" (width, height) dimension(s)
        - innermost indices are the features.

        :param features: TODO

        :return tuple of the inputs and labels
        """

        # https://stackoverflow.com/a/16816243
        # [0]     means line 0 of your matrix
        # [(0,0)] means cell at 0,0 of your matrix
        # [0:1]   means lines 0 to 1 excluded of your matrix
        # [:1]    excluding the first value means all lines until line 1 excluded
        # [1:]    excluding the last param mean all lines starting form line 1  included
        # [:]     excluding both means all lines
        # [::2]   the addition of a second ':' is the sampling. (1 item every 2)
        # [::]    exluding it means a sampling of 1
        # [:,:]   simply uses a tuple (a single , represents an empty tuple) instead of an index.

        inputs: tf.Tensor = features[:, self.input_slice, :]
        labels: tf.Tensor = features[:, self.label_slices, :]

        if self.label_columns:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        print("All shapes are: (batch, time, feature)")
        print(f"Window shape: {features.shape}")
        print(f"Inputs shape: {inputs.shape}")
        print(f"Labels shape: {labels.shape}")

        return inputs, labels

    def not_implemented(self) -> None:
        raise NotImplementedError()

    def printer(self) -> None:
        print(sys.version)
        print(requests.__version__)
        print(tf.__version__)
        print(pd.__version__)
        print(np.__version__)
        print(keras.__version__)
