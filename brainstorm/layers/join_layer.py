#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError, flatten_all_but_last

# FIXME
import numpy as np
import re

def Join(name=None):
    """TODO add documentation"""
    return ConstructionWrapper.create(JoinLayerImpl, name=name)


class JoinLayerImpl(Layer):
    expected_inputs = {
'input0': StructureTemplate('T', 'B', '...'),
'switch': StructureTemplate('T', 'B', 1)
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

    def setup(self, kwargs, in_shapes):
        in_shape = in_shapes['input0'].feature_shape
        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', *in_shape)
        parameters = OrderedDict()
        internals = OrderedDict()
        internals['stored_switch'] = BufferStructure('T', 'B', 1)
        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # TODO put this into handler, currently only supports numpy
        desired_inputs = np.unique(buffers.inputs.switch).tolist()
        buffers.internals.stored_switch[:] = buffers.inputs.switch

        buffers.outputs.default[:] = -np.inf
#         print('FORWARD: di are',desired_inputs,'available',buffers.inputs.keys())
        for ip in desired_inputs:
            where = (buffers.inputs.switch[:,:,0] == ip) # remove last dimension
            buffers.outputs.default[where,:] = buffers.inputs['input%d' % ip][where,:]

        assert np.all(np.isfinite(buffers.outputs.default))

            
    def backward_pass(self, buffers):
        desired_inputs = np.unique(buffers.internals.stored_switch).tolist()
        the_input_names = [x for x in buffers.input_deltas.keys() if re.match(r'input.*',x)]
        for ipn in the_input_names:
            buffers.input_deltas[ipn][:] = 0.0 # NOT inf, since these might remain 0
        for ip in desired_inputs:
            where = (buffers.internals.stored_switch[:,:,0] == ip) # remove last dimension
            buffers.input_deltas['input%d' % ip][where,:] = buffers.output_deltas.default[where,:]



