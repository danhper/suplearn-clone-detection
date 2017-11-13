import copy

from keras.layers.wrappers import Wrapper
from keras.utils.generic_utils import has_arg
import keras.backend as K


class SplitInput(Wrapper):
    def __init__(self, layer, weights=None, **kwargs):
        super(SplitInput, self).__init__(layer, **kwargs)
        self.forward_layer = copy.copy(layer)
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
        if weights:
            nw = len(weights)
            self.forward_layer.initial_weights = weights[:nw // 2]
            self.backward_layer.initial_weights = weights[nw // 2:]
        self.stateful = layer.stateful
        self.return_sequences = layer.return_sequences
        self.supports_masking = True

    def get_weights(self):
        return self.forward_layer.get_weights() + self.backward_layer.get_weights()

    def set_weights(self, weights):
        nw = len(weights)
        self.forward_layer.set_weights(weights[:nw // 2])
        self.backward_layer.set_weights(weights[nw // 2:])

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[1] //= 2
        output_shape = list(self.forward_layer.compute_output_shape(tuple(shape)))
        output_shape[1] *= 2
        return tuple(output_shape)

    def call(self, inputs, training=None, mask=None):
        kwargs = {}
        if has_arg(self.layer.call, 'training'):
            kwargs['training'] = training
        if has_arg(self.layer.call, 'mask'):
            kwargs['mask'] = mask

        ni = inputs.shape[1]
        y = self.forward_layer.call(inputs[:, :ni // 2], **kwargs)
        y_rev = self.backward_layer.call(inputs[:, ni // 2:], **kwargs)
        output = K.concatenate([y, y_rev])

        # Properly set learning phase
        if 0 < self.layer.dropout + self.layer.recurrent_dropout:
            output._uses_learning_phase = True
        return output

    def reset_states(self):
        self.forward_layer.reset_states()
        self.backward_layer.reset_states()

    def build(self, input_shape=None):
        with K.name_scope(self.forward_layer.name):
            self.forward_layer.build(input_shape)
        with K.name_scope(self.backward_layer.name):
            self.backward_layer.build(input_shape)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        if self.return_sequences:
            return mask
        else:
            return None

    @property
    def trainable_weights(self):
        if hasattr(self.forward_layer, 'trainable_weights'):
            return (self.forward_layer.trainable_weights +
                    self.backward_layer.trainable_weights)
        return []

    @property
    def non_trainable_weights(self):
        if hasattr(self.forward_layer, 'non_trainable_weights'):
            return (self.forward_layer.non_trainable_weights +
                    self.backward_layer.non_trainable_weights)
        return []

    @property
    def updates(self):
        if hasattr(self.forward_layer, 'updates'):
            return self.forward_layer.updates + self.backward_layer.updates
        return []

    @property
    def losses(self):
        if hasattr(self.forward_layer, 'losses'):
            return self.forward_layer.losses + self.backward_layer.losses
        return []

    @property
    def constraints(self):
        constraints = {}
        if hasattr(self.forward_layer, 'constraints'):
            constraints.update(self.forward_layer.constraints)
            constraints.update(self.backward_layer.constraints)
        return constraints
