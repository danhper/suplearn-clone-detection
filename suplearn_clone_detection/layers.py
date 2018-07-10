from typing import List
import copy
from os import path

from keras import activations, initializers, regularizers, constraints
from keras.engine import Layer, InputSpec, Model
from keras.layers.merge import _Merge, Dot
from keras.layers.wrappers import Wrapper
from keras.utils.generic_utils import has_arg
import keras.backend as K



class ModelWrapper(Model):
    @property
    def inner_models(self) -> List[Model]:
        return [v for v in self.layers if isinstance(v, Model)]

    def save(self, filepath, overwrite=True, include_optimizer=True):
        kwargs = dict(overwrite=overwrite, include_optimizer=include_optimizer)
        filename, ext = path.splitext(filepath)
        super(ModelWrapper, self).save(filepath, **kwargs)
        for model in self.inner_models:
            fullpath = f"{filename}-{model.name}{ext}"
            model.save(fullpath, **kwargs)

    def summary(self, line_length=None, positions=None, print_fn=print):
        kwargs = dict(line_length=line_length, positions=positions,
                      print_fn=print_fn)
        for model in self.inner_models:
            print_fn(model.name)
            model.summary(**kwargs)
        print("Main model:")
        super(ModelWrapper, self).summary(**kwargs)


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
        if self.layer.dropout + self.layer.recurrent_dropout > 0:
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
        constr = {}
        if hasattr(self.forward_layer, 'constraints'):
            constr.update(self.forward_layer.constraints)
            constr.update(self.backward_layer.constraints)
        return constr


class _PairMerge(_Merge):
    def _check_merge_inputs(self, inputs):
        class_name = self.__class__.__name__
        if len(inputs) != 2:
            raise ValueError('`{0}` layer should be called '
                             'on exactly 2 inputs'.format(class_name))
        if K.int_shape(inputs[0]) != K.int_shape(inputs[1]):
            raise ValueError('`{0}` layer should be called '
                             'on inputs of the same shape'.format(class_name))


class AbsDiff(_PairMerge):
    def _merge_function(self, inputs):
        self._check_merge_inputs(inputs)
        return K.abs(inputs[0] - inputs[1])


class EuclideanDistance(_PairMerge):
    def __init__(self, *args, max_value=None, normalize=False, **kwargs):
        super(EuclideanDistance, self).__init__(*args, **kwargs)
        self.max_value = max_value
        self.normalize = normalize
        if self.normalize and not self.max_value:
            raise ValueError("max_value must be provided to normalize output")

    def _merge_function(self, inputs):
        self._check_merge_inputs(inputs)
        distance = norm(inputs[0] - inputs[1])
        if self.max_value:
            distance = K.clip(distance, 0, self.max_value)
        if self.normalize:
            distance /= self.max_value
        return distance

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        return (shape[0], 1)


class EuclideanSimilarity(EuclideanDistance):
    def __init__(self, *args, max_value=None, **kwargs):
        super(EuclideanSimilarity, self).__init__(
            *args, max_value=max_value, normalize=True, **kwargs)
        if not max_value:
            raise ValueError("max_value must be provided")

    def _merge_function(self, inputs):
        distance = super(EuclideanSimilarity, self)._merge_function(inputs)
        return 1 - distance


class CosineSimilarity(Dot):
    def __init__(self, *args, min_value=-1, **kwargs):
        # set default axis to 1
        if not args and "axes" not in kwargs:
            args = (1,)
        super(CosineSimilarity, self).__init__(*args, **kwargs)
        self.min_value = min_value

    def call(self, inputs):
        dot_product = super(CosineSimilarity, self).call(inputs)
        magnitude = norm(inputs[0]) * norm(inputs[1])
        result = dot_product / magnitude
        return K.clip(result, self.min_value, 1)

    def get_config(self):
        config = {"min_value": self.min_value}
        base_config = super(CosineSimilarity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def norm(x):
    return K.sqrt(K.sum(K.pow(x, 2)))


def abs_diff(inputs, **kwargs):
    return AbsDiff(**kwargs)(inputs)


def euclidean_distance(inputs, **kwargs):
    return EuclideanDistance(**kwargs)(inputs)


def euclidean_similarity(inputs, **kwargs):
    return EuclideanSimilarity(**kwargs)(inputs)


def cosine_similarity(inputs, **kwargs):
    return CosineSimilarity(**kwargs)(inputs)


class DenseMulti(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseMulti, self).__init__(**kwargs)
        self.units = units
        self.kernels = []
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = []
        self.supports_masking = False
        self.bias = None

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('`DenseMulti` layer should be called '
                             'on a list of inputs')
        assert len(input_shape) >= 2

        for i, shape in enumerate(input_shape):
            assert len(shape) == 2
            assert shape[0] == input_shape[0][0]

            input_dim = shape[-1]

            self.kernels.append(self.add_weight(shape=(input_dim, self.units),
                                                initializer=self.kernel_initializer,
                                                name='kernel-{0}'.format(i),
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint))
            self.input_spec.append(InputSpec(min_ndim=2, axes={-1: input_dim}))
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        self.built = True

    def call(self, inputs):
        assert len(inputs) == len(self.kernels)

        output = K.dot(inputs[0], self.kernels[0])
        for i in range(1, len(inputs)):
            output += K.dot(inputs[i], self.kernels[i])
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('`DenseMulti` layer should be called '
                             'on a list of inputs')
        for shape in input_shape:
            assert len(shape) == 2
            assert shape[0] == input_shape[0][0]
        assert input_shape[0][-1]
        output_shape = list(input_shape[0])
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseMulti, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    "SplitInput": SplitInput,
    "AbsDiff": AbsDiff,
    "DenseMulti": DenseMulti,
    "ModelWrapper": ModelWrapper,
    "CosineSimilarity": CosineSimilarity,
    "EuclideanSimilarity": EuclideanSimilarity,
}
