#
#  model_test.py
#
#  Copyright 2024 Pepe Unlimited
#  Licensed under the MIT license, see associated LICENSE file for terms.
#  See AUTHORS file for the list of project authors.
#

import unittest
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from src.time_series_forecasting import window_generator as tsf
from src.time_series_forecasting import dataset as ds
from src.time_series_forecasting import model as m
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as max
import seaborn as sns


class TestModel(unittest.TestCase):

    # @unittest.skip
    def test_1h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()

        wg = tsf.WindowGenerator(
            input_width=24,
            label_width=24,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=["T (degC)"],
        )

        # test
        for inputs, labels in wg.train.take(1):
            # print(f"Inputs shape (batch, time, features): {inputs.shape}")
            # print(f"Labels shape (batch, time, features): {labels.shape}")
            self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
            self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, 1])

        model = m.Model(label_index=wg.column_indices["T (degC)"])
        model.compile(
            loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanAbsoluteError()]
        )

        val_performance = {}
        performance = {}
        val_performance["Baseline"] = model.evaluate(wg.val, return_dict=True)
        performance["Baseline"] = model.evaluate(wg.test, verbose="auto", return_dict=True)

        # inputs, labels and output
        inputs, labels = wg.example
        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, 1])
        self.assertListEqual(np.array(model(inputs).shape).tolist(), [32, 24, 1])  # output
        # print("Input shape:", inputs.shape)
        # print("Output shape:", baseline(inputs).shape)
        # print(labels[0, :, 0][23])
        # print(model(inputs)[0, :, 0][23])
        wg.plot(
            inputs=inputs,
            labels=labels,
            model=model,
            fname="1h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/1h_into_future_24h_history.png"
            )
        )

    # @unittest.skip
    def test_linear_1h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()

        wg = tsf.WindowGenerator(
            input_width=24,
            label_width=24,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=["T (degC)"],
        )

        # test
        for inputs, labels in wg.train.take(1):
            # print(f"Inputs shape (batch, time, features): {inputs.shape}")
            # print(f"Labels shape (batch, time, features): {labels.shape}")
            self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
            self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, 1])

        # linear model
        linear = keras.Sequential([keras.layers.Dense(units=1)])
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
        linear.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        history = linear.fit(
            wg.train, epochs=20, validation_data=wg.val, callbacks=[early_stopping]
        )

        # layer weight
        plt.bar(x=range(len(wg.train_df.columns)), height=linear.layers[0].kernel[:, 0].numpy())
        axis = plt.gca()
        axis.set_xticks(range(len(wg.train_df.columns)))
        _ = axis.set_xticklabels(wg.train_df.columns, rotation=90)
        plt.savefig(fname=f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/layer_weight_bar_plot.png")
        plt.close()

        # performance
        val_performance = {}
        performance = {}
        val_performance["Linear"] = linear.evaluate(wg.val, return_dict=True)
        performance["Linear"] = linear.evaluate(wg.test, verbose="auto", return_dict=True)

        # inputs, labels and output
        inputs, labels = wg.example
        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, 1])
        self.assertListEqual(np.array(linear(inputs).shape).tolist(), [32, 24, 1])  # output
        wg.plot(
            inputs=inputs,
            labels=labels,
            model=linear,
            fname="linear_1h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/linear_1h_into_future_24h_history.png"
            )
        )

    # @unittest.skip
    def test_stack_dense_1h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()

        wg = tsf.WindowGenerator(
            input_width=24,
            label_width=24,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=["T (degC)"],
        )

        # test
        for inputs, labels in wg.train.take(1):
            # print(f"Inputs shape (batch, time, features): {inputs.shape}")
            # print(f"Labels shape (batch, time, features): {labels.shape}")
            self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
            self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, 1])

        # stack dense model
        stack_dense = keras.Sequential(
            [
                keras.layers.Dense(units=64, activation="relu"),
                keras.layers.Dense(units=64, activation="relu"),
                keras.layers.Dense(units=1),
            ]
        )
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
        stack_dense.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        history = stack_dense.fit(
            wg.train, epochs=20, validation_data=wg.val, callbacks=[early_stopping]
        )

        # performance
        val_performance = {}
        performance = {}
        val_performance["Dense"] = stack_dense.evaluate(wg.val, return_dict=True)
        performance["Dense"] = stack_dense.evaluate(wg.test, verbose="auto", return_dict=True)

        # inputs, labels and output
        inputs, labels = wg.example
        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, 1])
        self.assertListEqual(np.array(stack_dense(inputs).shape).tolist(), [32, 24, 1])  # output
        wg.plot(
            inputs=inputs,
            labels=labels,
            model=stack_dense,
            fname="stack_dense_1h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/stack_dense_1h_into_future_24h_history.png"
            )
        )

    # @unittest.skip
    def test_multi_dense_1h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()

        wg = tsf.WindowGenerator(
            input_width=3,
            label_width=1,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=["T (degC)"],
        )

        # test
        for inputs, labels in wg.train.take(1):
            # print(f"Inputs shape (batch, time, features): {inputs.shape}")
            # print(f"Labels shape (batch, time, features): {labels.shape}")
            self.assertListEqual(np.array(inputs.shape).tolist(), [32, 3, 19])
            self.assertListEqual(np.array(labels.shape).tolist(), [32, 1, 1])

        # You could train a dense model on a multiple-input-step window by adding
        # a tf.keras.layers.Flatten as the first layer of the model
        #
        # stack_dense_multi_step model

        stack_dense_multi_step = keras.Sequential(
            [
                # Shape: (time, features) => (time*features)
                keras.layers.Flatten(),
                keras.layers.Dense(units=32, activation="relu"),
                keras.layers.Dense(units=32, activation="relu"),
                keras.layers.Dense(units=1),
                # Add back the time dimension.
                # Shape: (outputs) => (1, outputs)
                keras.layers.Reshape([1, -1]),
            ]
        )

        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
        stack_dense_multi_step.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        history = stack_dense_multi_step.fit(
            wg.train, epochs=20, validation_data=wg.val, callbacks=[early_stopping]
        )

        # performance
        val_performance = {}
        performance = {}
        val_performance["Dense"] = stack_dense_multi_step.evaluate(wg.val, return_dict=True)
        performance["Dense"] = stack_dense_multi_step.evaluate(
            wg.test, verbose="auto", return_dict=True
        )

        # inputs, labels and output
        inputs, labels = wg.example
        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 3, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 1, 1])
        self.assertListEqual(
            np.array(stack_dense_multi_step(inputs).shape).tolist(), [32, 1, 1]
        )  # output
        wg.plot(
            inputs=inputs,
            labels=labels,
            model=stack_dense_multi_step,
            fname="multi_dense_1h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/multi_dense_1h_into_future_24h_history.png"
            )
        )

    # @unittest.skip
    def test_convolution_neural_network_1h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()

        conv_width = 3
        label_width = 24
        input_width = label_width + (conv_width - 1)

        wg = tsf.WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=["T (degC)"],
        )

        # test

        for inputs, labels in wg.train.take(1):
            # print(f"Inputs shape (batch, time, features): {inputs.shape}")
            # print(f"Labels shape (batch, time, features): {labels.shape}")
            self.assertListEqual(np.array(inputs.shape).tolist(), [32, 26, 19])
            self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, 1])

        # A convolution layer (tf.keras.layers.Conv1D) also takes multiple time
        # steps as input to each prediction.
        #
        # Below is the same model as multi_step_dense, re-written with a convolution.
        #
        # Note the changes:
        #
        # The tf.keras.layers.Flatten and the first tf.keras.layers.Dense are replaced by a
        # tf.keras.layers.Conv1D.
        #
        # The tf.keras.layers.Reshape is no longer necessary since the convolution keeps the time
        # axis in its output.

        conv_model = keras.Sequential(
            [
                keras.layers.Conv1D(filters=32, kernel_size=(conv_width,), activation="relu"),
                keras.layers.Dense(units=32, activation="relu"),
                keras.layers.Dense(units=1),
            ]
        )

        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
        conv_model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        history = conv_model.fit(
            wg.train, epochs=20, validation_data=wg.val, callbacks=[early_stopping]
        )

        # performance
        val_performance = {}
        performance = {}
        val_performance["Conv"] = conv_model.evaluate(wg.val, return_dict=True)
        performance["Conv"] = conv_model.evaluate(wg.test, verbose="auto", return_dict=True)

        # inputs, labels and outputs
        inputs, labels = wg.example

        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 26, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, 1])
        self.assertListEqual(np.array(conv_model(inputs).shape).tolist(), [32, 24, 1])  # output

        # Note that the output is shorter than the input. To make training or plotting work,
        # you need the labels, and prediction to have the same length. So build a WindowGenerator
        # to produce wide windows with a few extra input time steps so the label and
        # prediction lengths match.

        wg.plot(
            inputs=inputs,
            labels=labels,
            model=conv_model,
            fname="conv_1h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/conv_1h_into_future_24h_history.png"
            )
        )

    # @unittest.skip
    def test_rnn_lstm_multi_feature_1h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()

        # The models so far all predicted a single output feature, T (degC), for a single time step.
        #
        # All of these models can be converted to predict multiple features just by changing the
        # number of units in the output layer and adjusting the training windows to include all
        # features in the labels (example_labels):
        #
        # `WindowGenerator` returns all features as labels if you don't set the `label_columns` argument.

        num_features = 19

        wg = tsf.WindowGenerator(
            input_width=24,
            label_width=24,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        for inputs, labels in wg.train.take(1):
            # print(f"Inputs shape (batch, time, features): {inputs.shape}")
            # print(f"Labels shape (batch, time, features): {labels.shape}")
            self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
            self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])

        # If False, the default, the layer only returns the output of the final time step, giving
        # the model time to warm up its internal state before making a single prediction
        #
        # If True, the layer returns an output for each input. This is useful for:
        # Stacking RNN layers.
        # Training a model on multiple time steps simultaneously.
        #
        # With return_sequences=True, the model can be trained on 24 hours of data at a time.

        lstm_model = keras.models.Sequential(
            [
                # Shape [batch, time, features] => [batch, time, lstm_units]
                keras.layers.LSTM(32, return_sequences=True),
                keras.layers.Dense(units=num_features),
            ]
        )

        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
        lstm_model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        history = lstm_model.fit(
            wg.train, epochs=20, validation_data=wg.val, callbacks=[early_stopping]
        )

        # cm = lstm_model.metrics[1]
        # print(cm.metrics)

        # performance
        val_performance = {}
        performance = {}
        val_performance["LSTM"] = lstm_model.evaluate(wg.val, return_dict=True)
        performance["LSTM"] = lstm_model.evaluate(wg.test, verbose="auto", return_dict=True)

        x = np.arange(len(performance))
        width = 0.3
        metric_name = "mean_absolute_error"
        val_mae = [v[metric_name] for v in val_performance.values()]
        test_mae = [v[metric_name] for v in performance.values()]

        plt.ylabel("mean_absolute_error [T (degC), normalized]")
        plt.bar(x - 0.17, val_mae, width, label="Validation")
        plt.bar(x + 0.17, test_mae, width, label="Test")
        plt.xticks(ticks=x, labels=performance.keys(), rotation=45)
        _ = plt.legend()
        plt.savefig(fname=f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/performance_plot.png")
        plt.close()

        for name, value in performance.items():
            print(f"{name:12s}: {value[metric_name]:0.4f}")

        # inputs, labels and outputs
        inputs, labels = wg.example

        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])
        self.assertListEqual(
            np.array(lstm_model(inputs).shape).tolist(), [32, 24, num_features]
        )  # output

        wg.plot(
            inputs=inputs,
            labels=labels,
            model=lstm_model,
            fname="rnn_lstm_multi_feature_1h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/rnn_lstm_multi_feature_1h_into_future_24h_history.png"
            )
        )

    def test_residual_lstm_multi_feature_1h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()
        num_features = 19

        wg = tsf.WindowGenerator(
            input_width=24,
            label_width=24,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        for inputs, labels in wg.train.take(1):
            # print(f"Inputs shape (batch, time, features): {inputs.shape}")
            # print(f"Labels shape (batch, time, features): {labels.shape}")
            self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
            self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])

        residual_lstm = m.ResidualWrapper(
            keras.models.Sequential(
                [
                    keras.layers.LSTM(32, return_sequences=True),
                    keras.layers.Dense(
                        units=num_features, kernel_initializer=keras.initializers.zeros()
                    ),
                ]
            )
        )

        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
        residual_lstm.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        history = residual_lstm.fit(
            wg.train, epochs=20, validation_data=wg.val, callbacks=[early_stopping]
        )
        # inputs, labels and outputs
        inputs, labels = wg.example

        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])
        self.assertListEqual(
            np.array(residual_lstm(inputs).shape).tolist(), [32, 24, num_features]
        )  # output

        wg.plot(
            inputs=inputs,
            labels=labels,
            model=residual_lstm,
            fname="residual_lstm_multi_feature_1h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/residual_lstm_multi_feature_1h_into_future_24h_history.png"
            )
        )

    def test_baseline_24h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()

        out_steps = 24
        num_features = 19
        # `WindowGenerator` returns all features as labels if you don't set the `label_columns`
        # argument.
        wg = tsf.WindowGenerator(
            input_width=24,
            label_width=24,
            shift=out_steps,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        for inputs, labels in wg.train.take(1):
            # print(f"Inputs shape (batch, time, features): {inputs.shape}")
            # print(f"Labels shape (batch, time, features): {labels.shape}")
            self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
            self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])

        last_baseline = m.MultiStepLastBaseline(out_steps=out_steps)
        last_baseline.compile(
            loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanAbsoluteError()]
        )

        # inputs, labels and outputs
        inputs, labels = wg.example

        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])
        self.assertListEqual(
            np.array(last_baseline(inputs).shape).tolist(), [32, 24, num_features]
        )  # output

        wg.plot(
            inputs=inputs,
            labels=labels,
            model=last_baseline,
            fname="baseline_24h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/baseline_24h_into_future_24h_history.png"
            )
        )

    def test_repeat_24h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()

        out_steps = 24
        num_features = 19
        # `WindowGenerator` returns all features as labels if you don't set the `label_columns`
        # argument.
        wg = tsf.WindowGenerator(
            input_width=24,
            label_width=24,
            shift=out_steps,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        for inputs, labels in wg.train.take(1):
            # print(f"Inputs shape (batch, time, features): {inputs.shape}")
            # print(f"Labels shape (batch, time, features): {labels.shape}")
            self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
            self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])

        repeat_baseline = m.RepeatBaseline()
        repeat_baseline.compile(
            loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanAbsoluteError()]
        )

        # inputs, labels and outputs
        inputs, labels = wg.example

        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])
        self.assertListEqual(
            np.array(repeat_baseline(inputs).shape).tolist(), [32, 24, num_features]
        )  # output

        wg.plot(
            inputs=inputs,
            labels=labels,
            model=repeat_baseline,
            fname="repeat_24h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/repeat_24h_into_future_24h_history.png"
            )
        )

    def test_linear_24h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()

        out_steps = 24
        num_features = 19
        # `WindowGenerator` returns all features as labels if you don't set the `label_columns`
        # argument.
        wg = tsf.WindowGenerator(
            input_width=24,
            label_width=24,
            shift=out_steps,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        for inputs, labels in wg.train.take(1):
            # print(f"Inputs shape (batch, time, features): {inputs.shape}")
            # print(f"Labels shape (batch, time, features): {labels.shape}")
            self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
            self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])

        linear_model = keras.Sequential(
            [
                # shape [batch, time, features] => [batch, 1, features]
                keras.layers.Lambda(lambda x: x[:, -1, :]),
                # shape => [batch, 1, out_steps * features]
                keras.layers.Dense(
                    out_steps * num_features, kernel_initializer=keras.initializers.zeros()
                ),
                # shape => [batch, out_steps, features]
                keras.layers.Reshape([out_steps, num_features]),
            ]
        )

        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
        linear_model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        _ = linear_model.fit(
            wg.train, epochs=20, validation_data=wg.val, callbacks=[early_stopping]
        )

        # inputs, labels and outputs
        inputs, labels = wg.example

        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])
        self.assertListEqual(
            np.array(linear_model(inputs).shape).tolist(), [32, 24, num_features]
        )  # output

        wg.plot(
            inputs=inputs,
            labels=labels,
            model=linear_model,
            fname="linear_24h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/linear_24h_into_future_24h_history.png"
            )
        )

    def test_multi_dense_24h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()

        out_steps = 24
        num_features = 19
        # `WindowGenerator` returns all features as labels if you don't set the `label_columns`
        # argument.
        wg = tsf.WindowGenerator(
            input_width=24,
            label_width=24,
            shift=out_steps,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        for inputs, labels in wg.train.take(1):
            # print(f"Inputs shape (batch, time, features): {inputs.shape}")
            # print(f"Labels shape (batch, time, features): {labels.shape}")
            self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
            self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])

        multi_dense = keras.Sequential(
            [
                # shape [batch, time, features] => [batch, 1, features]
                keras.layers.Lambda(lambda x: x[:, -1, :]),
                # shape => [batch, 1, dense_units]
                keras.layers.Dense(512, activation="relu"),
                # shape => [batch, 1, out_steps * features]
                keras.layers.Dense(
                    out_steps * num_features, kernel_initializer=keras.initializers.zeros()
                ),
                # shape => [batch, out_steps, features]
                keras.layers.Reshape([out_steps, num_features]),
            ]
        )

        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
        multi_dense.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        _ = multi_dense.fit(wg.train, epochs=20, validation_data=wg.val, callbacks=[early_stopping])

        # inputs, labels and outputs
        inputs, labels = wg.example

        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])
        self.assertListEqual(
            np.array(multi_dense(inputs).shape).tolist(), [32, 24, num_features]
        )  # output

        wg.plot(
            inputs=inputs,
            labels=labels,
            model=multi_dense,
            fname="multi_dense_24h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/multi_dense_24h_into_future_24h_history.png"
            )
        )

    def test_cnn_24h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()

        out_steps = 24
        num_features = 19
        conv_width = 3

        # `WindowGenerator` returns all features as labels if you don't set the `label_columns`
        # argument.
        wg = tsf.WindowGenerator(
            input_width=24,
            label_width=24,
            shift=out_steps,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        for inputs, labels in wg.train.take(1):
            # print(f"Inputs shape (batch, time, features): {inputs.shape}")
            # print(f"Labels shape (batch, time, features): {labels.shape}")
            self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
            self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])

        cnn = keras.Sequential(
            [
                # shape [batch, time, features] => [batch, CONV_WIDTH, features]
                keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),
                # shape => [batch, 1, conv_units]
                keras.layers.Conv1D(256, activation="relu", kernel_size=(conv_width)),
                # shape => [batch, 1, out_steps * features]
                keras.layers.Dense(
                    out_steps * num_features, kernel_initializer=keras.initializers.zeros()
                ),
                # shape => [batch, out_steps, features]
                keras.layers.Reshape([out_steps, num_features]),
            ]
        )

        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
        cnn.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        _ = cnn.fit(wg.train, epochs=20, validation_data=wg.val, callbacks=[early_stopping])

        # inputs, labels and outputs
        inputs, labels = wg.example

        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])
        self.assertListEqual(np.array(cnn(inputs).shape).tolist(), [32, 24, num_features])  # output

        wg.plot(
            inputs=inputs,
            labels=labels,
            model=cnn,
            fname="cnn_24h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/cnn_24h_into_future_24h_history.png"
            )
        )

    def test_lstm_24h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()

        out_steps = 24
        num_features = 19

        # `WindowGenerator` returns all features as labels if you don't set the `label_columns`
        # argument.
        wg = tsf.WindowGenerator(
            input_width=24,
            label_width=24,
            shift=out_steps,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        for inputs, labels in wg.train.take(1):
            # print(f"Inputs shape (batch, time, features): {inputs.shape}")
            # print(f"Labels shape (batch, time, features): {labels.shape}")
            self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
            self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])

        lstm = keras.Sequential(
            [
                # Shape [batch, time, features] => [batch, lstm_units].
                # Adding more `lstm_units` just overfits more quickly.
                keras.layers.LSTM(32, return_sequences=False),
                # Shape => [batch, out_steps * features]
                keras.layers.Dense(
                    out_steps * num_features,
                    kernel_initializer=keras.initializers.zeros(),
                ),
                # Shape => [batch, out_steps, features].
                keras.layers.Reshape([out_steps, num_features]),
            ]
        )

        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
        lstm.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        _ = lstm.fit(wg.train, epochs=20, validation_data=wg.val, callbacks=[early_stopping])

        # inputs, labels and outputs
        inputs, labels = wg.example

        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])
        self.assertListEqual(
            np.array(lstm(inputs).shape).tolist(), [32, 24, num_features]
        )  # output

        wg.plot(
            inputs=inputs,
            labels=labels,
            model=lstm,
            fname="lstm_24h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/lstm_24h_into_future_24h_history.png"
            )
        )

    def test_lstm_autoregressive_24h_into_future_24h_history(self):
        train_df, val_df, test_df = ds.weather_dataset()

        out_steps = 24
        num_features = 19
        units = 32

        feedback_model: m.Feedback = m.Feedback(
            units=units, out_steps=out_steps, num_features=num_features
        )

        # `WindowGenerator` returns all features as labels if you don't set the `label_columns`
        # argument.
        wg = tsf.WindowGenerator(
            input_width=24,
            label_width=24,
            shift=out_steps,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        # print(f"Output shape (batch, time, features): {feedback_model(wg.example[0]).shape}")
        self.assertListEqual(np.array(feedback_model(wg.example[0]).shape).tolist(), [32, 24, 19])

        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
        feedback_model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        _ = feedback_model.fit(
            wg.train, epochs=20, validation_data=wg.val, callbacks=[early_stopping]
        )

        # inputs, labels and outputs
        inputs, labels = wg.example

        self.assertListEqual(np.array(inputs.shape).tolist(), [32, 24, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [32, 24, num_features])
        self.assertListEqual(
            np.array(feedback_model(inputs).shape).tolist(), [32, 24, num_features]
        )  # output

        wg.plot(
            inputs=inputs,
            labels=labels,
            model=feedback_model,
            fname="lstm_autoregressive_24h_into_future_24h_history",
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/lstm_autoregressive_24h_into_future_24h_history.png"
            )
        )


if __name__ == "__main__":
    unittest.main()
