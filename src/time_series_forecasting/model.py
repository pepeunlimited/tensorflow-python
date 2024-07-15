#
#  model.py
#
#  Copyright 2024 Pepe Unlimited
#  Licensed under the MIT license, see associated LICENSE file for terms.
#  See AUTHORS file for the list of project authors.
#

import keras
import tensorflow as tf


class Model(keras.Model):

    label_index: int

    def __init__(self, label_index: int = 0):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index == 0:
            return inputs

        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


class ResidualWrapper(keras.Model):
    """
    The Baseline model from earlier took advantage of the fact that the sequence doesn't change
    drastically from time step to time step. Every model trained in this tutorial so far was
    randomly initialized, and then had to learn that the output is a a small change from the
    previous time step.

    While you can get around this issue with careful initialization, it's simpler to build this
    into the model structure.

    It's common in time series analysis to build models that instead of predicting the next value,
    predict how the value will change in the next time step. Similarly, residual networks—or
    ResNets—in deep learning refer to architectures where each layer adds to the model's
    accumulating result.

    That is how you take advantage of the knowledge that the change should be small.
    """

    model: keras.models.Sequential

    def __init__(self, model: keras.models.Sequential):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each time step is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta


class MultiStepLastBaseline(keras.Model):
    """
    A simple baseline for this task is to repeat the last input time step for the required number
    # of output time steps.
    """

    out_steps: int

    def __init__(self, out_steps: int = 1):
        super().__init__()
        self.out_steps = out_steps

    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, self.out_steps, 1])


class RepeatBaseline(keras.Model):
    """
    Since this task is to predict 24 hours into the future, given 24 hours of the past, another
    simple approach is to repeat the previous day, assuming tomorrow will be similar.
    """

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs


class Feedback(keras.Model):
    """
    This tutorial only builds an autoregressive RNN model, but this pattern could be applied to
    any model that was designed to output a single time step.

    The model will have the same basic form as the single-step LSTM models from earlier:
    a tf.keras.layers.LSTM layer followed by a tf.keras.layers.Dense layer that converts the LSTM
    layer's outputs to model predictions.

    A tf.keras.layers.LSTM is a tf.keras.layers.LSTMCell wrapped in the higher level
    tf.keras.layers.RNN that manages the state and sequence results for you
    (Check out the Recurrent Neural Networks (RNN) with Keras guide for details).

    In this case, the model has to manually manage the inputs for each step, so it uses
    tf.keras.layers.LSTMCell directly for the lower level, single time step interface.
    """

    units: int
    out_steps: int
    num_features: int

    lstm_cell: keras.layers.LSTMCell
    lstm_rnn: keras.layers.RNN
    dense: keras.layers.Dense

    def __init__(self, units: int, out_steps: int, num_features: int):
        super().__init__()
        self.units = units
        self.out_steps = out_steps
        self.lstm_cell = keras.layers.LSTMCell(units)
        # also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = keras.layers.Dense(num_features)

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions: tf.TensorArray = []
        # initialize the LSTM state
        prediction, state = self.warmup(inputs)
        # insert the first prediction
        predictions.append(prediction)
        # run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # use the last prediction as input
            x = prediction
            # execute one lstm step
            x, state = self.lstm_cell(x, states=state, training=training)
            # convert the lstm output to a prediction
            prediction = self.dense(x)
            # add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state
