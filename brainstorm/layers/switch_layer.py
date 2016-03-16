#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError, flatten_all_but_last


def Switch(name=None):
    """TODO add documentation"""
    return ConstructionWrapper.create(SwitchLayerImpl, name=name)


class SwitchLayerImpl(Layer):
    expected_inputs = {
'input0': StructureTemplate('T', 'B', '...'),
'switch': StructureTemplate('T', 'B')
}

    optional_inputs = {
'input1': StructureTemplate('T', 'B', '...'),
'input2': StructureTemplate('T', 'B', '...'),
'input3': StructureTemplate('T', 'B', '...'),
'input4': StructureTemplate('T', 'B', '...'),
'input5': StructureTemplate('T', 'B', '...'),
'input6': StructureTemplate('T', 'B', '...'),
'input7': StructureTemplate('T', 'B', '...'),
'input8': StructureTemplate('T', 'B', '...'),
'input9': StructureTemplate('T', 'B', '...'),
}

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
        in_shape = in_shapes['input1'].feature_shape
        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', *in_shape)
        parameters = OrderedDict()
        internals = OrderedDict()
        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # TODO put this inot handler, currentyl only supports numpy
        desired_inputs = np.unique(buffers.inputs.switch).tolist()

        buffers.outputs.default = np.full(buffers.inputs.input1.shape,-np.inf)
        for ip in desired_inputs:
            where = (buffers.inputs.switch == ip)
            buffers.outputs.default[where] = buffers.inputs.input1[where]

            
    def backward_pass(self, buffers):
        # prepare
        self.handler.reverse_with_mask(buffers.output_deltas.default,
                buffers.inputs.mask, buffers.input_deltas.default)


