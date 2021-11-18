#!/usr/bin/env python3

from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant


class ContinuousLayer(Layer):
    """
    Costume Layer for the training with continuous actionspace. The layers input should be the policy (mu) to get the
    same shape size. The variance is saved as trainable weights.
    """

    def __init__(self, variance=0.0, **kwargs):
        """
        Constructor of the ContinuousLayer.
        :param variance: The start value for the variance weights.
        :param kwargs: quarks arguments.
        """
        self._variance = None
        self._start_variance = variance
        super(ContinuousLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Overload build function from keras layer.
        :param input_shape: Input shape of the layer
        :return:
        """
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        # variance weights
        self._variance = self.add_weight(name='kernel',
                                         shape=(input_dim,),
                                         initializer=Constant(self._start_variance),
                                         trainable=True)

    def call(self, x, **kwargs):
        """
        Overload call funtion from keras layer.
        :param x: The input as the output from the previous layer.
        :param kwargs: quarks arguments.
        :return: layer output.
        """
        # I hate keras for this shit. Needed for the (None,...) shape. The x input will be ignored.
        return x * 0.0 + self._variance

    def get_config(self):
        """
        Overlad get_config function from keras.
        :return: config.
        """
        base_config = super(ContinuousLayer, self).get_config()
        return base_config
