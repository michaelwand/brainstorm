#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError, flatten_all_but_last


def Reverse(name=None):
    """Create a layer that merges two inputs into one along the last dim"""
    return ConstructionWrapper.create(ReverseLayerImpl, name=name)


class ReverseLayerImpl(Layer):
    expected_inputs = {'default': StructureTemplate('T', 'B', 'F')}

    optional_inputs = {'mask': StructureTemplate('T', 'B', 1)}

    expected_kwargs = {}

    # TODO this could become the standard implementation
    def _validate_in_shapes(self):
        """Ensure all in_shapes are valid by comparing to `expected_inputs`.

        Raises:
            LayerValidationError: if there are unrecognized inputs, missing
                                  inputs or inputs that don't match the
                                  `StructureTemplate` from `expected_inputs`.
        """
        in_shape_names = set(self.in_shapes.keys())
        input_names = set(self.expected_inputs.keys())
        optional_input_names = set(self.optional_inputs.keys())

        all_inputs = self.expected_inputs.copy()
        all_inputs.update(self.optional_inputs)

        if not in_shape_names.issubset(input_names | optional_input_names):
            raise LayerValidationError(
                'Invalid in_shapes. {} has no input(s) named "{}". Choices '
                'are: {}'.format(self.name, in_shape_names - input_names,
                                 input_names))

        if not input_names.issubset(in_shape_names):
            raise LayerValidationError(
                '{}: All required inputs need to be connected. Missing {}.'
                .format(self.name, input_names - in_shape_names))

        for input_name, in_shape in self.in_shapes.items():
            if not all_inputs[input_name].matches(in_shape):
                raise LayerValidationError(
                    "{}: in_shape ({}) for {} doesn't match StructureTemplate "
                    "{}".format(self.name, in_shape, input_name,all_inputs[input_name]))
    def setup(self, kwargs, in_shapes):
        in_shape = in_shapes['default'].feature_size
        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', in_shape)
        parameters = OrderedDict()
        internals = OrderedDict()
        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        self.handler.reverse_with_mask(buffers.inputs.default,
                buffers.inputs.mask, buffers.outputs.default)

    def backward_pass(self, buffers):
        # prepare
        self.handler.reverse_with_mask(buffers.output_deltas.default,
                buffers.inputs.mask, buffers.input_deltas.default)

