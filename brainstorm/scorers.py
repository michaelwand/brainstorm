#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

import numpy as np

from brainstorm.describable import Describable

from brainstorm.layers.ctc_layer import ctc_greedy_decoding, ctc_token_passing_decoding, levenshtein

# ----------------------------- Base Class ---------------------------------- #

class Scorer(Describable):
    def __init__(self, out_name='', targets_name='targets', mask_name='',
                 name=None):
        self.out_name = out_name
        self.targets_name = targets_name
        self.mask_name = mask_name
        self.__name__ = name if name is not None else self.__class__.__name__

    def __call__(self, true_labels, predicted, mask=None):
        pass

    @staticmethod
    def aggregate(errors):
        errors = np.array(errors)
        assert errors.ndim == 2 and errors.shape[1] == 2
        return np.sum(errors[:, 1]) / np.sum(errors[:, 0])


# ------------------------- Scoring Functions ------------------------------- #

def gather_losses_and_scores(net, scorers, scores, out_name='',
                             targets_name='targets', mask_name=''):
    ls = net.get_loss_values()
    for name, loss in ls.items():
        scores[name].append((net._buffer_manager.batch_size, loss))

    for sc in scorers:
        name = sc.__name__
        predicted = net.get(sc.out_name or out_name or net.output_name)
        true_labels = net.get_input(sc.targets_name) if sc.targets_name\
            else net.get_input(targets_name)
        mask = net.get_input(sc.mask_name) if sc.mask_name\
            else (net.get_input(mask_name) if mask_name else None)

        weight = mask.sum() if mask is not None else predicted.shape[0] * predicted.shape[1]

        # TODO a hack
        this_result = sc(true_labels, predicted, mask) 
        if isinstance(this_result,tuple):
            weight = this_result[0]
            this_result = this_result[1]
        scores[name].append((weight, this_result))
# #         scores[name].append((weight, sc(true_labels, predicted, mask)))


def aggregate_losses_and_scores(scores, net, scorers):
    results = OrderedDict()
    for name in net.get_loss_values():
        results[name] = _weighted_average(scores[name])
    for sc in scorers:
        results[sc.__name__] = sc.aggregate(scores[sc.__name__])
    return results


# ------------------------------- Scorers ----------------------------------- #

class Accuracy(Scorer):
    def __call__(self, true_labels, predicted, mask=None):
        predicted = _flatten_all_but_last(predicted)
        true_labels = _flatten_all_but_last(true_labels)
        if mask is not None:
            mask = _flatten_all_but_last(mask)
    
        if predicted.shape[1] > 1:
            predicted = predicted.argmax(1).reshape(-1, 1)
        correct = (predicted == true_labels).astype(np.float)
        if mask is not None:
            correct *= mask
        return np.sum(correct)


class Hamming(Scorer):
    def __init__(self, threshold=0.5, out_name='', targets_name='targets',
                 mask_name='', name=None):
        super(Hamming, self).__init__(out_name, targets_name, mask_name, name)
        self.threshold = threshold

    def __call__(self, true_labels, predicted, mask=None):
        predicted = _flatten_all_but_last(predicted)
        true_labels = _flatten_all_but_last(true_labels)
        if mask is not None:
            mask = _flatten_all_but_last(mask)
        correct = np.logical_xor(predicted < self.threshold,
                                 true_labels).astype(np.float)
        if mask is not None:
            correct *= mask
        return np.sum(correct) / true_labels.shape[1]


class MeanSquaredError(Scorer):
    def __call__(self, true_labels, predicted, mask=None):
        predicted = _flatten_all_but_last(predicted)
        true_labels = _flatten_all_but_last(true_labels)
        if mask is not None:
            mask = _flatten_all_but_last(mask)
        errors = (true_labels - predicted) ** 2
        if mask is not None:
            errors *= mask
        return 0.5 * np.sum(errors)

class LabelingError(Scorer):
    def __init__(self, out_name='', targets_name='labels', mask_name='',
                 decoding_method = 'greedy',name=None):
        super(LabelingError, self).__init__(out_name, targets_name, mask_name, name)
        if decoding_method == 'greedy':
            self.decoding_function = ctc_greedy_decoding
#         elif decoding_method == 'best_path':
#             self.decoding_function = ctc_best_path_decoding
        else:
            raise Exception('Unimplemented decoding method for CTC LabelingError')

    def __call__(self, true_labels, predicted, mask=None):
        assert true_labels.ndim == 3
        assert true_labels.shape[2] == 1

        assert predicted.ndim == 3
        assert true_labels.shape[1] == predicted.shape[1]

        true_labels = true_labels.astype(int)
        # TODO note: if this were framewise decoding, would maximize over ax2 of predicted here
        # also note that we must apply the mask, and that true_labels will probably contain zeros (blanks)

        errors = 0
        total_length = 0

        for sequence in range(true_labels.shape[1]):
            if mask is not None:
                this_mask = mask[:,sequence,0].astype(bool)
                these_predictions = predicted[this_mask,sequence,:]
            else:
                these_predictions = predicted[:,sequence,:]

            # run decoding
            ctc_prediction = self.decoding_function(these_predictions) # TODO add extra params for language model etc.

            # compare to labels (with blanks removed)
            these_true_labels = true_labels[:,sequence,0]
            assert these_true_labels.ndim == 1
            label_zeros = np.where(these_true_labels == 0)[0]

            if np.any(label_zeros):
                label_boundary = np.min(label_zeros)
                # TODO whcih exception?
                if not np.all(these_true_labels[label_boundary:] == 0): 
                    raise Exception('Label with non-contiguous zeros passed to CTC')

                these_true_labels = these_true_labels[0:label_boundary]

# # # # #             print('CTC decoding: ',these_true_labels,' ---> ',ctc_prediction)
            errors += levenshtein(ctc_prediction,these_true_labels)
            total_length += len(these_true_labels)

        return (total_length,errors) # total_length serves as weight for error aggregation

    # should be able to use standard aggregate method




# ---------------------------- Helper Functions ----------------------------- #

def _flatten_all_but_last(a):
    if a is None:
        return None
    return a.reshape(-1, a.shape[-1])


def _weighted_average(errors):
    errors = np.array(errors)
    assert errors.ndim == 2 and errors.shape[1] == 2
    return np.sum(errors[:, 1] * errors[:, 0] / np.sum(errors[:, 0]))
