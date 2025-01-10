from typing import Dict

from keras import backend
from keras import layers
from keras import models
import tensorflow as tf

from acp.layers import autoregressive_gru, gumbel_softmax
from acp.models.decoders import decoder
from acp.models import model


class ACPDecoder(decoder.Decoder):

    def __init__(
            self,
            latent_dim: int,
            dense: layers.Dense,
            recurrent_autoregressive: autoregressive_gru.AutoregressiveGRU,
            lstm: layers.LSTM,
            activation: gumbel_softmax.GumbelSoftmax,
            name: str = 'ACPsExpandedDecoder'
    ):
        self.latent_dim = latent_dim
        self.latent_to_hidden = recurrent_autoregressive
        self.dense = dense
        self.lstm = lstm
        self.activation = activation
        self.name = name
        self.concatenation_reminder_1 = layers.Concatenate(name='decoder_condition_remainder_1')
        self.concatenation_reminder_2 = layers.Concatenate(name='decoder_condition_remainder_2')
        self.get_condition_from_input = layers.Lambda(lambda x_: x_[:, -2:], name="decoder_get_condition_from_input")
        self.distribute_condition_remainder = layers.RepeatVector(25)
        self.time_distribute_activation = layers.TimeDistributed(self.activation,
                                                                 name='decoder_time_distribute_activation')

    def output_tensor(self, input_=None):
        x = input_
        c = self.get_condition_from_input(x)
        c = self.distribute_condition_remainder(c)
        x = self.call_layer_on_input(self.latent_to_hidden, x)
        x = self.concatenation_reminder_1([x, c])
        x = self.call_layer_on_input(self.lstm, x)
        x = self.concatenation_reminder_2([x, c])
        x = self.call_layer_on_input(self.dense, x)
        return self.call_layer_on_input(self.time_distribute_activation, x)

    def __call__(self, input_=None):
        z = input_
        generated_x = self.output_tensor(z)
        model = models.Model(z, generated_x)
        return model

    def get_config_dict(self) -> Dict:
        return {
            'type': type(self).__name__,
            'name': self.name,
            'latent_dim': self.latent_dim,
            'output_dim': self.latent_to_hidden.output_dim,
            'output_len': self.latent_to_hidden.output_len,
        }

    def get_layers_with_names(self) -> Dict[str, layers.Layer]:
        return {
            f'{self.name}_latent_to_hidden': self.latent_to_hidden.recurrent,
            f'{self.name}_lstm': self.lstm,
            f'{self.name}_dense': self.dense,
        }

    @classmethod
    def from_config_dict_and_layer_collection(
            cls,
            config_dict: Dict,
            layer_collection: model.ModelLayerCollection,
    ) -> "ACPDecoder":
        recurrent_autoregressive = autoregressive_gru.AutoregressiveGRU(
            output_dim=config_dict['output_dim'],
            output_len=config_dict['output_len'],
            recurrent=layer_collection[config_dict['name'] + '_latent_to_hidden'],
        )
        return cls(
            name=config_dict['name'],
            latent_dim=config_dict['latent_dim'],
            recurrent_autoregressive=recurrent_autoregressive,
            dense=layer_collection[config_dict['name'] + '_dense'],
            lstm=layer_collection[config_dict['name'] + '_lstm'],
            activation=gumbel_softmax.GumbelSoftmax()
        )


class ACPDecoderFactory:

    @staticmethod
    def build_default(
            latent_dim: int,
            gumbel_temperature: tf.Variable,
            max_length: int,
    ):
        recurrent_autoregressive = autoregressive_gru.AutoregressiveGRU.build_for_gru(
            latent_dim,
            max_length,
        )
        lstm = layers.LSTM(
            100,
            unroll=True,
            stateful=False,
            dropout=0.1,
            return_sequences=True,
            name="decoder_lstm"
        )

        dense = layers.Dense(21, name="decoder_output")
        return ACPDecoder(
            latent_dim=latent_dim,
            recurrent_autoregressive=recurrent_autoregressive,
            lstm=lstm,
            dense=dense,
        )
