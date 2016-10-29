#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError, flatten_all_but_last


def Merge(name=None):
    """Create a layer that merges two inputs into one along the last dim"""
    return ConstructionWrapper.create(MergeLayerImpl, name=name)


class MergeLayerImpl(Layer):
    expected_inputs = {
            'input0': StructureTemplate('...'),
            }
    optional_inputs = {
            'input1': StructureTemplate('...'),
            'input2': StructureTemplate('...'),
            'input3': StructureTemplate('...'),
            'input4': StructureTemplate('...'),
            'input5': StructureTemplate('...'),
            'input6': StructureTemplate('...'),
            'input7': StructureTemplate('...'),
            'input8': StructureTemplate('...'),
            'input9': StructureTemplate('...'),
            }
    expected_kwargs = {}

    def setup(self, kwargs, in_shapes):
        self.in_shape_names = sorted(set(in_shapes.keys()))

        # All inputs must have the same shape except for last dim
        shape_prefix_0 = in_shapes['input0'].shape[:-1]

        for in_shape in self.in_shape_names:
            this_shape_prefix = in_shapes[in_shape].shape[:-1]
            if shape_prefix_0 != this_shape_prefix:
                raise LayerValidationError(
                    "{}: The shapes of inputs 0 and {} may only differ in the last dimension but got {} and {}".format(
                        self.name, in_shape, in_shapes['input0'].shape, in_shapes[in_shape].shape))

        combined_size = sum([ in_shapes[k].shape[-1] for k in self.in_shape_names ])

        out_shape = shape_prefix_0 + (combined_size,)
        outputs = OrderedDict()
        outputs['default'] = BufferStructure(*out_shape)

        parameters = OrderedDict()
        internals = OrderedDict()
        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        self.handler.multimerge_t([buffers.inputs[x] for x in self.in_shape_names], buffers.outputs.default)

    def backward_pass(self, buffers):
        self.handler.multisplit_add_t(buffers.output_deltas.default,[buffers.input_deltas[x] for x in self.in_shape_names])
#         # prepare
#         _h = self.handler
#         raise Exception('This does not work')
#         self.handler.split_add_tt(buffers.output_deltas.default,
#                                   buffers.input_deltas.inputs_1,
#                                   buffers.input_deltas.inputs_2)
