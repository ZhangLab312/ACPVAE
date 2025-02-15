from typing import Any, Dict, Optional, List

import tensorflow as tf
from acp.layers import vae_loss
from acp.models import model as acp_model
from acp.models.decoders import acp_expanded_decoder
from acp.models.decoders import decoder as dec
from acp.models.discriminators import acp_classifier_noCONV
from acp.models.discriminators import discriminator as disc
from acp.models.discriminators import veltri_acp_classifier
from acp.models.encoders import acp_expanded_encoder
from acp.models.encoders import encoder as enc
from acp.utils import metrics
from keras import backend as K
from keras import layers, models, optimizers, losses


class MasterACPTrainer(acp_model.Model):

    def __init__(
            self,
            encoder: enc.Encoder,
            decoder: dec.Decoder,
            acp_classifier: disc.Discriminator,
            mic_classifier: disc.Discriminator,
            kl_weight: float,
            rcl_weight: int,
            master_optimizer: optimizers.Optimizer,
            loss_weights: Optional[List[float]],
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.acp_classifier = acp_classifier
        self.mic_classifier = mic_classifier
        self.kl_weight = kl_weight
        self.rcl_weight = rcl_weight
        self.master_optimizer = master_optimizer
        self.loss_weights = loss_weights

    @staticmethod
    def sampling(input_: Optional[Any] = None):
        noise_in, z_mean, z_sigma = input_
        return z_mean + K.exp(z_sigma / 2) * noise_in

    def build(self, input_shape: Optional):
        if self.loss_weights is None:
            raise AttributeError("Please set loss weight before training. Configs can be found at acp.config")
        self.acp_classifier.freeze_layers()
        self.mic_classifier.freeze_layers()

        sequences_input = layers.Input(shape=(input_shape[0],), name="sequences_input")
        z_mean, z_sigma, z = self.encoder.output_tensor(sequences_input)
        mic_in = layers.Input(shape=(1,), name="mic_in")
        acp_in = layers.Input(shape=(1,), name="acp_in")
        sleep_mic_in = layers.Input(shape=(1,), name="sleep_mic_in")
        sleep_acp_in = layers.Input(shape=(1,), name="sleep_acp_in")
        noise_in = layers.Input(shape=(64,), name="noise_in")

        z = layers.Lambda(self.sampling, output_shape=(64,), name="z")
        z = z([noise_in, z_mean, z_sigma])
        z_cond = layers.concatenate([z, acp_in, mic_in], name="z_cond")

        reconstructed = self.decoder.output_tensor(z_cond)
        acp_output = self.acp_classifier.output_tensor_with_dense_input(input_=reconstructed)
        mic_output = self.mic_classifier.output_tensor_with_dense_input(input_=reconstructed)
        z_cond_reconstructed = self.encoder.output_tensor_with_dense_input(reconstructed)[0]
        z_cond_reconstructed_error = layers.Subtract(name="z_cond_reconstructed_error")([z, z_cond_reconstructed])
        # end of cvae
        sleep_z_cond = layers.concatenate([z, sleep_acp_in, sleep_mic_in], name='sleep_z_cond')
        sleep_reconstructed = self.decoder.output_tensor(sleep_z_cond)
        sleep_cond_reconstructed = self.encoder.output_tensor_with_dense_input(sleep_reconstructed)[0]
        sleep_cond_reconstructed_error = layers.Subtract(name="correction_sleep_cond_reconstructed_error")(
            [z, sleep_cond_reconstructed])

        unconstrained_sleep_z_cond = layers.concatenate([noise_in, sleep_acp_in, sleep_mic_in],
                                                        name="unconstrained_sleep_z_cond")
        unconstrained_sleep_reconstructed = self.decoder.output_tensor(unconstrained_sleep_z_cond)
        unconstrained_sleep_cond_reconstructed = \
            self.encoder.output_tensor_with_dense_input(unconstrained_sleep_reconstructed)[0]
        unconstrained_sleep_cond_reconstructed_error = layers.subtract(
            [noise_in, unconstrained_sleep_cond_reconstructed], name="unconstrained_sleep_cond_reconstructed_error")

        sleep_acp_output = self.acp_classifier.output_tensor_with_dense_input(input_=sleep_reconstructed)
        unconstrained_sleep_acp_output = self.acp_classifier.output_tensor_with_dense_input(
            input_=unconstrained_sleep_reconstructed,
        )

        sleep_mic_output = self.mic_classifier.output_tensor_with_dense_input(input_=sleep_reconstructed)
        unconstrained_sleep_mic_output = self.mic_classifier.output_tensor_with_dense_input(
            input_=unconstrained_sleep_reconstructed,
        )

        def idn_f(x_):
            return x_

        acp_output_wrap = \
            layers.Lambda(idn_f, name="acp_prediction")(acp_output)
        mic_output_wrap = \
            layers.Lambda(idn_f, name="mic_prediction")(mic_output)

        correction_sleep_acp_output_wrap = \
            layers.Lambda(idn_f, name="correction_sleep_acp_prediction")(sleep_acp_output)
        correction_sleep_mic_output_wrap = \
            layers.Lambda(idn_f, name="correction_sleep_mic_prediction")(sleep_mic_output)

        unconstrained_sleep_acp_output_wrap = \
            layers.Lambda(idn_f, name="unconstrained_sleep_acp_prediction")(unconstrained_sleep_acp_output)
        unconstrained_sleep_mic_output_wrap = \
            layers.Lambda(idn_f, name="unconstrained_sleep_mic_prediction")(unconstrained_sleep_mic_output)

        mic_mean_grad = K.gradients(
            loss=mic_output,
            variables=[z_mean]
        )[0]

        acp_mean_grad = K.gradients(
            loss=acp_output,
            variables=[z_mean]
        )[0]

        mic_mean_grad_input = layers.Input(
            tensor=tf.math.scalar_mul(self.decoder.activation.temperature, mic_mean_grad),
            name="mic_mean_grad"
        )

        acp_mean_grad_input = layers.Input(
            tensor=tf.math.scalar_mul(self.decoder.activation.temperature, acp_mean_grad),
            name="acp_mean_grad"
        )

        unconstrained_sleep_mic_output_grad  = K.gradients(
            loss=unconstrained_sleep_mic_output,
            variables=[noise_in]
        )[0]

        unconstrained_sleep_acp_output_grad  = K.gradients(
            loss=unconstrained_sleep_acp_output,
            variables=[noise_in]
        )[0]

        unconstrained_sleep_mic_output_grad_input = layers.Input(
            tensor=tf.math.scalar_mul(self.decoder.activation.temperature, unconstrained_sleep_mic_output_grad),
            name="unconstrained_sleep_mic_output_grad_input"
        )

        unconstrained_sleep_acp_output_grad_input = layers.Input(
            tensor=tf.math.scalar_mul(self.decoder.activation.temperature, unconstrained_sleep_acp_output_grad),
            name="unconstrained_sleep_acp_output_grad_input"
        )
        # TODO: gradient w.r.t to input?

        correction_sleep_mic_output_grad = K.gradients(
            loss=sleep_mic_output,
            variables=[z_mean]
        )[0]

        correction_sleep_acp_output_grad = K.gradients(
            loss=sleep_acp_output,
            variables=[z_mean]
        )[0]

        correction_sleep_mic_output_grad_input = layers.Input(
            tensor=tf.math.scalar_mul(self.decoder.activation.temperature, correction_sleep_mic_output_grad),
            name="correction_sleep_mic_output_grad"
        )

        correction_sleep_acp_output_grad_input = layers.Input(
            tensor=tf.math.scalar_mul(self.decoder.activation.temperature, correction_sleep_acp_output_grad),
            name="correction_sleep_acp_output_grad"
        )

        y = vae_loss.VAELoss(
            kl_weight=self.kl_weight,
            rcl_weight=self.rcl_weight,
        )([sequences_input, reconstructed, z_mean, z_sigma])

        vae = models.Model(
            inputs=[
                sequences_input,
                acp_in,
                mic_in,
                noise_in,
                mic_mean_grad_input,
                acp_mean_grad_input,
                unconstrained_sleep_mic_output_grad_input,
                unconstrained_sleep_acp_output_grad_input,
                correction_sleep_mic_output_grad_input,
                correction_sleep_acp_output_grad_input,
                sleep_acp_in,
                sleep_mic_in,
            ],
            outputs=[
                acp_output_wrap,
                mic_output_wrap,
                y,
                mic_mean_grad_input,
                acp_mean_grad_input,
                unconstrained_sleep_mic_output_grad_input,
                unconstrained_sleep_acp_output_grad_input,
                correction_sleep_mic_output_grad_input,
                correction_sleep_acp_output_grad_input,
                correction_sleep_acp_output_wrap,
                correction_sleep_mic_output_wrap,
                unconstrained_sleep_acp_output_wrap,
                unconstrained_sleep_mic_output_wrap,
                z_cond_reconstructed_error,
                sleep_cond_reconstructed_error,
                unconstrained_sleep_cond_reconstructed_error,
            ]
        )

        kl_metric = metrics.kl_loss(z_mean, z_sigma)

        def _kl_metric(y_true, y_pred):
            return kl_metric

        reconstruction_acc = metrics.sparse_categorical_accuracy(sequences_input, reconstructed)

        def _reconstruction_acc(y_true, y_pred):
            return reconstruction_acc

        rcl = metrics.reconstruction_loss(sequences_input, reconstructed)

        def _rcl(y_true, y_pred):
            return rcl

        amino_acc, empty_acc = metrics.get_generation_acc()(sequences_input, reconstructed)

        def _amino_acc(y_true, y_pred):
            return amino_acc

        def _empty_acc(y_true, y_pred):
            return empty_acc

        def entropy(y_true, y_pred):
            return K.log(y_pred + K.epsilon()) * y_pred + K.log(1 - y_pred + K.epsilon()) * (1 - y_pred)

        def entropy_smoothed_loss(y_true, y_pred):
            return K.binary_crossentropy(y_true, y_pred) + 0.1 * entropy(y_true, y_pred)

        vae.compile(
            optimizer='adam',
            loss=[
                entropy_smoothed_loss,
                entropy_smoothed_loss,
                'mae',  # reconstruction
                losses.Huber(),
                losses.Huber(),
                losses.Huber(),
                losses.Huber(),
                losses.Huber(),
                losses.Huber(),
                entropy_smoothed_loss,
                entropy_smoothed_loss,
                entropy_smoothed_loss,  #
                entropy_smoothed_loss,
                'mse',  #
                'mse',
            ],
            loss_weights=self.loss_weights,
            metrics=[
                ['acc', 'binary_crossentropy'],
                ['acc', 'binary_crossentropy'],
                [_kl_metric, _rcl, _reconstruction_acc, _amino_acc, _empty_acc],  # reconstruction
                ['mse', losses.Huber()],
                ['mse', losses.Huber()],
                ['mse', losses.Huber()],
                ['mse', losses.Huber()],
                ['mse', losses.Huber()],  #
                ['mse', losses.Huber()],
                ['acc', 'binary_crossentropy'],
                ['acc', 'binary_crossentropy'],
                ['acc', 'binary_crossentropy'],
                ['acc', 'binary_crossentropy'],
                ['mse', 'mae'],
                ['mse', 'mae'],
                ['mse', 'mae'],

            ]
        )
        return vae

    def get_config_dict(self) -> Dict:
        return {
            'type': type(self).__name__,
            'encoder_config_dict': self.encoder.get_config_dict(),
            'decoder_config_dict': self.decoder.get_config_dict(),
            'acp_config_dict': self.acp_classifier.get_config_dict(),
            'mic_config_dict': self.mic_classifier.get_config_dict(),
        }

    @classmethod
    def from_config_dict_and_layer_collection(
            cls,
            config_dict: Dict,
            layer_collection: acp_model.ModelLayerCollection,
    ) -> "MasterACPTrainer":
        return cls(
            encoder=acp_expanded_encoder.ACPEncoder.from_config_dict_and_layer_collection(
                config_dict=config_dict['encoder_config_dict'],
                layer_collection=layer_collection,
            ),
            decoder=acp_expanded_decoder.ACPDecoder.from_config_dict_and_layer_collection(
                config_dict=config_dict['decoder_config_dict'],
                layer_collection=layer_collection,
            ),
            acp_classifier=acp_classifier_noCONV.NoConvACPClassifier.from_config_dict_and_layer_collection(
                config_dict=config_dict['acp_config_dict'],
                layer_collection=layer_collection,
            ),
            mic_classifier=veltri_acp_classifier.VeltriACPClassifier.from_config_dict_and_layer_collection(
                config_dict=config_dict['mic_config_dict'],
                layer_collection=layer_collection,
            ),

            kl_weight=K.variable(0.1),
            rcl_weight=32,
            master_optimizer=optimizers.Adam(lr=1e-3),
            loss_weights=None
        )

    def get_layers_with_names(self) -> Dict[str, layers.Layer]:
        layers_with_names = {}
        for name, layer in self.encoder.get_layers_with_names().items():
            layers_with_names[name] = layer
        for name, layer in self.decoder.get_layers_with_names().items():
            layers_with_names[name] = layer
        for name, layer in self.acp_classifier.get_layers_with_names().items():
            layers_with_names[name] = layer
        for name, layer in self.mic_classifier.get_layers_with_names().items():
            layers_with_names[name] = layer

        return layers_with_names
