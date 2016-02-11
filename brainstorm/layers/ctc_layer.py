#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import (LayerValidationError, flatten_all_but_last)

import pdb # TODO
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

    IMPORTANT WARNING:
        This layer currently considers the input data mask to be boolean (i.e. no
        weights are allowed), and it should have the form 1 ... 1 0 ... 0 (only the final
        unbroken sequence of zeros is removed)

    WARNING:
        This layer does not compute derivatives wrt the 'targets' input.
        It also does not use the deltas coming in from the 'predictions'.
    """
    return ConstructionWrapper.create(CTCLayerImpl, name=name)

# "invert" means that the path with smallest values through ctc_probabilities is found
def ctc_greedy_decoding(ctc_probabilities,invert = False):
    assert ctc_probabilities.ndim == 2 # no multibatch
    ctc_prediction = ctc_probabilities.argmax(axis=1) if not invert else ctc_probabilities.argmin(axis=1)

    result = []
    last_was_blank = True # also for start
    for pos in range(ctc_prediction.shape[0]):
        if last_was_blank and ctc_prediction[pos] != 0:
            result.append(ctc_prediction[pos])
            last_was_blank = False
        elif last_was_blank is False:
            if ctc_prediction[pos] == 0:
                last_was_blank = True
            elif ctc_prediction[pos] != result[-1]:
                result.append(ctc_prediction[pos])
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
                       'labels': StructureTemplate('T', 'B', 1)}

    optional_inputs = {'mask': StructureTemplate('T', 'B', 1)}

    computes_no_input_deltas_for = ['mask','labels']
    takes_no_output_deltas_from = ['predictions']

    def setup(self, kwargs, in_shapes):
        in_shape = in_shapes['default'].feature_size

        self.clip_ctc = kwargs.get('clip_ctc',np.float64(1e-20))
        self.use_warpctc = kwargs.get('use_warpctc',False)
        self.labels_on_gpu = kwargs.get('labels_on_gpu',False)

        outputs = OrderedDict()
        outputs['predictions'] = BufferStructure('T', 'B', in_shape)
        outputs['loss'] = BufferStructure('B', 1) 

        internals = OrderedDict()
        internals['temp_dinputs'] = BufferStructure('T', 'B', in_shape)
        internals['temp_dinputs2'] = BufferStructure('T', 'B', in_shape) # FIXME
        internals['softmax_deriv'] = BufferStructure('T', 'B', in_shape,
                                             is_backward_only=True)
        return outputs, OrderedDict(), internals 

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

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        inputs = buffers.inputs.default
        labels = buffers.inputs.labels.astype(np.int32)
        if 'mask' in buffers.inputs.keys():
            mask = buffers.inputs.mask.astype(int)
        else:
            mask = None
        predictions = buffers.outputs.predictions
        loss = buffers.outputs.loss

        temp_dinputs = buffers.internals.temp_dinputs

        # reshape
        flat_inputs = flatten_all_but_last(inputs)
        flat_predictions = flatten_all_but_last(predictions)

        # softmax - also needed for WarpCTC because of the predictions
        _h.softmax_m(flat_inputs, flat_predictions)

        # At this point, softmax is computed, and CTC code begins.

        if self.use_warpctc:
            if self.labels_on_gpu:
                raise Exception('TODO')
            _h.calculate_warpctc(inputs,mask,labels,temp_dinputs,loss,self.clip_ctc)

        else: 
            # The variable predictions has already been correctly filled
            # Here, we compute the loss, and we save the deltas to temp_dinputs
            # for being used in the backward pass.
            # All this is performed sequence-wise and currently does not parallelize.
            for sequence in xrange(inputs.shape[1]):
                if mask is not None:
                    this_mask = mask[:,sequence,0] # TODO: astype OK?
                    mask_zero_index = _h.get_final_zeros_index_v(this_mask) 

                    these_inputs = inputs[0:mask_zero_index,sequence,:]
                    these_predictions = predictions[0:mask_zero_index,sequence,:]
                else:
                    these_inputs = inputs[:,sequence,:]
                    these_predictions = predictions[:,sequence,:]

                these_uncut_labels = labels[:,sequence,0]

                final_zero_index = _h.get_final_zeros_index_v(these_uncut_labels)
                these_cut_labels = these_uncut_labels[0:final_zero_index]

                these_deltas = _h.allocate(these_predictions.shape)
    #             these_deltas2 = _h.allocate(these_predictions.shape)
#                 if self.use_warpctc:
#                     # TODO might pass entire minibatch
#                     # CPU <-> GPU argh
#                     if self.labels_on_gpu:
#                         this_error = _h.calculate_warpctc(these_inputs,these_cut_labels,these_deltas,self.clip_ctc)
#                     else:
#                         these_cut_labels_cpu = _h.get_numpy_copy(these_cut_labels)
#                         this_error = _h.calculate_warpctc(these_inputs,these_cut_labels_cpu,these_deltas,self.clip_ctc)
#     #                     this_error2 = _h.calculate_ctc(these_predictions,these_cut_labels_cpu,these_deltas2,self.clip_ctc)
#                 else:
                if 1:
                    this_error = _h.calculate_ctc(these_predictions,these_cut_labels,these_deltas,self.clip_ctc)
                    _h.mult_st(-1, these_deltas, these_deltas) # fold "minus one" into calculate_ctc?

    #             print('CTC LAYER: deltas b/w %f and %f' % (np.min(these_deltas), np.max(these_deltas)))

                # TODO annoying: single float no work on GPU
                loss[sequence,0] = np.array(this_error,dtype=loss.dtype)

                if mask is not None:
                    print('Temp_dinputs BEFORE is max %f' % np.max(abs(temp_dinputs[:,sequence,:].get())))
                    if np.max(abs(temp_dinputs[:,sequence,:].get())) > 0.001:
                        pass
                    temp_dinputs[0:mask_zero_index,sequence,:] = these_deltas
    #                 temp_dinputs2[0:mask_zero_index,sequence,:] = these_deltas2
                    _h.fill(temp_dinputs[mask_zero_index:,sequence,:],0)
    #                 temp_dinputs2[mask_zero_index:,sequence,:] = 0
                else:
                    temp_dinputs[:,sequence,:] = these_deltas
    #                 temp_dinputs2[:,sequence,:] = these_deltas2

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        predictions = buffers.outputs.predictions

        dinputs = buffers.input_deltas.default
        dloss = buffers.output_deltas.loss
        temp_dinputs = buffers.internals.temp_dinputs
#         temp_dinputs2 = buffers.internals.temp_dinputs2
        softmax_deriv = buffers.internals.softmax_deriv
#         softmax_deriv2 = np.empty_like(softmax_deriv)

        # reshape
        flat_predictions = flatten_all_but_last(predictions)
        flat_softmax_deriv = flatten_all_but_last(softmax_deriv)
#         flat_softmax_deriv2 = flatten_all_but_last(softmax_deriv2)
        flat_dloss = flatten_all_but_last(dloss)
        flat_dinputs = flatten_all_but_last(dinputs)
        flat_temp_dinputs = flatten_all_but_last(temp_dinputs)
#         flat_temp_dinputs2 = flatten_all_but_last(temp_dinputs2)

        # general softmax derivative
        if self.use_warpctc:
            _h.copy_to(flat_temp_dinputs,flat_softmax_deriv)
#             _h.softmax_deriv_m(flat_predictions,flat_temp_dinputs2,flat_softmax_deriv2)
        else:
            _h.softmax_deriv_m(flat_predictions,flat_temp_dinputs,flat_softmax_deriv)

        # Multiply with sequencewise loss.
        # Multiplication requires "manual broadcasting" so that it works with the PyCuda handler.
        for time in range(softmax_deriv.shape[0]):
            sub_softmax_deriv = softmax_deriv[time,:,:]
            _h.mult_mv(sub_softmax_deriv, flat_dloss, sub_softmax_deriv)

        _h.add_tt(flat_softmax_deriv, flat_dinputs, flat_dinputs)

#         print('CTC LAYER BACKWARD: SM between %f and %f, dinputs between %f and %f' % 
#                 (np.min(flat_softmax_deriv),np.max(flat_softmax_deriv),np.min(flat_dinputs),np.max(flat_dinputs)))


