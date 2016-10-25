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
        in_shape_names = set(in_shapes.keys())

        # All inputs must have the same shape except for last dim
        shape_prefix_0 = in_shapes['input0'].shape[:-1]

        for in_shape in in_shape_names:
            this_shape_prefix = in_shapes[in_shape].shape[:-1]
            if shape_prefix1 != shape_prefix2:
                raise LayerValidationError(
                    "{}: The shapes of inputs 0 and {} may only differ in the last dimension but got {} and {}".format(
                        self.name, in_shape, in_shapes['input0'].shape, in_shapes[in_shape].shape))

        combined_size = sum([ in_shapes[k].shape[-1] for k in in_shape_names ])

        out_shape = shape_prefix_0 + (combined_size,)
        outputs = OrderedDict()
        outputs['default'] = BufferStructure(*out_shape)

        parameters = OrderedDict()
        internals = OrderedDict()
        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        raise Exception('This does not work')
        self.handler.merge_tt(buffers.inputs.inputs_1,
                              buffers.inputs.inputs_2,
                              buffers.outputs.default)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        raise Exception('This does not work')
        self.handler.split_add_tt(buffers.output_deltas.default,
                                  buffers.input_deltas.inputs_1,
                                  buffers.input_deltas.inputs_2)
