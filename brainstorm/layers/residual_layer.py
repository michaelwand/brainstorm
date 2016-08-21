#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError
from brainstorm.utils import (LayerValidationError, flatten_time,
                              flatten_time_and_features)

def ResidualLayer(size, name=None):
    """Create a Residual layer."""
    return ConstructionWrapper.create(ResidualLayerImpl, name=name, size=size)


class ResidualLayerImpl(Layer):
    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'size'}

    def setup(self, kwargs, in_shapes):
        self.size = kwargs.get('size', self.in_shapes['default'].feature_size)

        outputs = OrderedDict()
        outputs['default'] = BufferStructure(
            'T', 'B', *self.in_shapes['default'].feature_shape)
        parameters = OrderedDict()
        parameters['W_rel'] = BufferStructure(self.size, self.size)
        parameters['bias_rel'] = BufferStructure(self.size)
        parameters['W_lin'] = BufferStructure(self.size, self.size)
        parameters['bias_lin'] = BufferStructure(self.size)
        internals = OrderedDict()
        internals['H1'] = BufferStructure('T', 'B', self.size)
        internals['H2'] = BufferStructure('T', 'B', self.size)
        internals['dH1'] = BufferStructure('T', 'B', self.size, is_backward_only=True)
        internals['dH2'] = BufferStructure('T', 'B', self.size, is_backward_only=True)
        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W_rel, bias_rel, W_lin, bias_lin = buffers.parameters
        inputs = flatten_time_and_features(buffers.inputs.default)
        outputs = flatten_time_and_features(buffers.outputs.default)
        H1, H2, dH1, dH2 = buffers.internals
        H1_flat = flatten_time_and_features(H1)
        H2_flat = flatten_time_and_features(H2)

        _h.dot_mm(inputs, W_rel, H1_flat, transb=True)
        _h.add_mv(H1_flat, bias_rel.reshape((1, bias_rel.shape[0])), H1_flat)
        _h.inplace_act_func['rel'](H1_flat)

        _h.dot_mm(H1_flat, W_lin, H2_flat, transb=True)
        _h.add_mv(H2_flat, bias_lin.reshape((1, bias_lin.shape[0])), H2_flat)
        _h.add_tt(H2_flat, inputs, out=outputs)
#         _h.copy_to(H2_flat, outputs)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler

        W_rel, bias_rel, W_lin, bias_lin = buffers.parameters
        dW_rel, dbias_rel, dW_lin, dbias_lin = buffers.gradients
        inputs = flatten_time_and_features(buffers.inputs.default)
        outputs = flatten_time_and_features(buffers.outputs.default)
        
        in_deltas = flatten_time_and_features(buffers.input_deltas.default)
        out_deltas = flatten_time_and_features(buffers.output_deltas.default)
        
        H1, H2, dH1, dH2 = buffers.internals
        H1_flat = flatten_time_and_features(H1)
        H2_flat = flatten_time_and_features(H2)
        dH1_flat = flatten_time_and_features(dH1)
        dH2_flat = flatten_time_and_features(dH2)

        # first part: linear layer
        _h.copy_to(out_deltas, dH2_flat)
        _h.dot_mm(dH2_flat, W_lin, dH1_flat)
        _h.dot_mm(dH2_flat, H1_flat, out=dW_lin, transa=True)
        _h.sum_t(dH2_flat, axis=0, out= dbias_lin)


        # second part: res layer

        _h.inplace_act_func_deriv['rel'](H1_flat, dH1_flat)
        _h.dot_add_mm(dH1_flat, W_rel, out=in_deltas) # in_deltas, first branch
        _h.add_st(1.0,in_deltas,out=in_deltas) # in_deltas, skip branch
        _h.dot_mm(dH1_flat, inputs, out=dW_rel, transa=True)
        _h.sum_t(dH1_flat, axis=0, out=dbias_rel)

