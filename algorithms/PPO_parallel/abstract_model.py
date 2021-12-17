from abc import ABC, abstractmethod
from tensorflow.keras import Model as KerasModel
import tensorflow as tf
import re
import datetime

class AbstractModel(ABC):
    _model: KerasModel

    @abstractmethod
    def __init__(self, config):
        self._name = type(self).__name__
        self._start_time = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")

    @abstractmethod
    def build(self):
        pass

    # @abstractmethod
    # def predict(self, obersavtion):
    #     pass

    @abstractmethod
    def train(self, observation, action):
        pass

    def save(self, steps):
        print('Saving....')
        self._model.save(f'saved_models/{self._name}_{self._start_time}/{self._name}_{self._start_time}_{steps}')

    def summary(self):
        self._model.summary()

    @classmethod
    def load_model(cls, path: str):
        obj = cls(None)
        obj._model = tf.keras.models.load_model(path)
        return obj

    @staticmethod
    def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):
        """
        Source: https://stackoverflow.com/a/54517478
        CC BY-SA 4.0
        """
        # Auxiliary dictionary to describe the network graph
        network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

        # Set the input layers of each layer
        for layer in model.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in network_dict['input_layers_of']:
                    network_dict['input_layers_of'].update(
                            {layer_name: [layer.name]})
                else:
                    network_dict['input_layers_of'][layer_name].append(layer.name)

        # Set the output tensor of the input layer
        network_dict['new_output_tensor_of'].update(
                {model.layers[0].name: model.input})

        # Iterate over all layers after the input
        model_outputs = []
        for layer in model.layers[1:]:

            # Determine input tensors
            layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                    for layer_aux in network_dict['input_layers_of'][layer.name]]
            if len(layer_input) == 1:
                layer_input = layer_input[0]

            # Insert layer if name matches the regular expression
            if re.match(layer_regex, layer.name):
                if position == 'replace':
                    x = layer_input
                elif position == 'after':
                    x = layer(layer_input)
                elif position == 'before':
                    pass
                else:
                    raise ValueError('position must be: before, after or replace')

                new_layer = insert_layer_factory()
                if insert_layer_name:
                    new_layer.name = insert_layer_name
                else:
                    new_layer.name = '{}_{}'.format(layer.name, 
                                                    new_layer.name)
                x = new_layer(x)
                print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                                layer.name, position))
                if position == 'before':
                    x = layer(x)
            else:
                x = layer(layer_input)

            # Set new output tensor (the original one, or the one of the inserted
            # layer)
            network_dict['new_output_tensor_of'].update({layer.name: x})

            # Save tensor in output list if it is output in initial model
            if layer_name in model.output_names:
                model_outputs.append(x)

        return KerasModel(inputs=model.inputs, outputs=model_outputs)
