#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import (LayerValidationError, flatten_all_but_last)

import sys

import numpy as np

def CTC(name=None):
    """Create a CTC layer which integrates a softmax loss function and the CTC algorithm.

    Applies the softmax activation function on 'default' input and puts
    results (per-class probabilities) in 'predictions'. It then takes sequence labels
    from the 'targets' input and uses them to compute the CTC loss, which is stored in
    the 'loss' output (indexed by sequence, but not by time). Suitable deltas are
    computed and backpropagated towards the 'default' input.

    Note that the labels must be
    - in the range 1 ... # of classes *inclusive*. 0 may not be used as a label since 
      it stands for the CTC 'blank' node.
    - of the size (time, batchsize, 1). This is a technical requirement of brainstorm.
      Where the label sequence is shorter, zeros are to be used for padding the time axis.

    Consequently, the layer size passed to the function bs.tools.get_in_out_layers_for_ctc
    (if you create your system that way) must be # of classes + 1.

    WARNING:
        This layer does not compute derivatives wrt the 'targets' input.
        It also does not use the deltas coming in from the 'predictions'.
    """
    return ConstructionWrapper.create(CTCLayerImpl, name=name)

def ctc_greedy_decoding(ctc_prediction):
    assert ctc_prediction.ndim == 2 # no multibatch
    ctc_max = ctc_prediction.argmax(axis=1)

    result = []
    last_was_blank = True # also for start
    for pos in range(ctc_max.shape[0]):
        if last_was_blank and ctc_max[pos] != 0:
            result.append(ctc_max[pos])
            last_was_blank = False
        elif last_was_blank is False:
            if ctc_max[pos] == 0:
                last_was_blank = True
            elif ctc_max[pos] != result[-1]:
                result.append(ctc_max[pos])
    return result 

    
def ctc_token_passing_decoding(ctc_prediction):
    raise Exception('Not implemented yet') # TODO

def levenshtein(seq1, seq2):
    oneago = None
    thisrow = range(1, len(seq2) + 1) + [0]
    for x in xrange(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in xrange(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
    return thisrow[len(seq2) - 1]

class CTCLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', 'F'),
                       'mask': StructureTemplate('T', 'B', 1),
                       'labels': StructureTemplate('T', 'B', 1)}

    computes_no_input_deltas_for = ['mask','labels']
    takes_no_output_deltas_from = ['predictions']

    def setup(self, kwargs, in_shapes):
        in_shape = in_shapes['default'].feature_size

        outputs = OrderedDict()
        outputs['predictions'] = BufferStructure('T', 'B', in_shape)
        outputs['loss'] = BufferStructure('B', 1) 

        internals = OrderedDict()
        internals['temp_dinputs'] = BufferStructure('T', 'B', in_shape)
        internals['softmax_deriv'] = BufferStructure('T', 'B', in_shape,
                                             is_backward_only=True)
        return outputs, OrderedDict(), internals 

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        inputs = buffers.inputs.default
        labels = buffers.inputs.labels
        mask = buffers.inputs.mask
        predictions = buffers.outputs.predictions
        loss = buffers.outputs.loss

        temp_dinputs = buffers.internals.temp_dinputs

        # reshape
        flat_inputs = flatten_all_but_last(inputs)
        flat_probs = flatten_all_but_last(predictions)

        # softmax
        _h.softmax_m(flat_inputs, flat_probs)

        # At this point, softmax is computed, and CTC code begins.
        # The variable predictions has already been correctly filled
        # Here, we compute the loss, and we save the deltas to temp_dinputs
        # for being used in the backward pass.
        # All this is performed sequence-wise and currently does not parallelize.
        for sequence in xrange(inputs.shape[1]):
            this_mask = mask[:,sequence,0].astype(bool) # TODO: astype OK?
            these_predictions = predictions[this_mask,sequence,:]
            these_uncut_labels = labels[:,sequence,0].astype(np.int64)

            final_zero_index = _h.get_final_zeros_index_v(these_uncut_labels)
            these_cut_labels = these_uncut_labels[0:final_zero_index]

            these_deltas = _h.allocate(these_predictions.shape)
            this_error = _h.calculate_ctc(these_predictions,these_cut_labels,these_deltas)
            _h.mult_st(-1, these_deltas, these_deltas) # fold - into calculate_ctc?

            loss[sequence,0] = this_error
            temp_dinputs[this_mask.astype(bool),sequence,:] = these_deltas

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        probs = buffers.outputs.predictions

        dinputs = buffers.input_deltas.default
        dloss = buffers.output_deltas.loss
        temp_dinputs = buffers.internals.temp_dinputs
        softmax_deriv = buffers.internals.softmax_deriv

        # reshape
        flat_probs = flatten_all_but_last(probs)
        flat_softmax_deriv = flatten_all_but_last(softmax_deriv)
        flat_dloss = flatten_all_but_last(dloss)
        flat_dinputs = flatten_all_but_last(dinputs)
        flat_temp_dinputs = flatten_all_but_last(temp_dinputs)

        # general softmax derivative
        _h.softmax_deriv_m(flat_probs,flat_temp_dinputs,flat_softmax_deriv)

        # Multiply with sequencewise loss.
        # Multiplication requires "manual broadcasting" so that it works with the PyCuda handler.
        for time in range(softmax_deriv.shape[0]):
            sub_softmax_deriv = softmax_deriv[time,:,:]
            _h.mult_mv(sub_softmax_deriv, flat_dloss, sub_softmax_deriv)

        _h.add_tt(flat_softmax_deriv, flat_dinputs, flat_dinputs)



