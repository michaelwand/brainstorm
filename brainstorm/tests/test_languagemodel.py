from __future__ import division, print_function, unicode_literals

import numpy as np
import pytest
import datetime

import sys
sys.path.insert(1,'/home/mwand/software/Brainstorm-Devel/brainstorm')

from brainstorm import languagemodel

language_model_file = '/home/mwand/software/Brainstorm-Devel/Bigram.002-101.SRI.LM.test1'
# large_language_model_file = '/home/mwand/software/Brainstorm-Devel/BNnTr96.arpabo.3'
# large_language_model_file = '/home/mwand/software/Brainstorm-Devel/MiniModel'
large_language_model_file = '/home/mwand/software/Brainstorm-Devel/Bigram.002-101.SRI.LM.test1'

def test_language_model_results():
    example_lm = languagemodel.LanguageModel(language_model_file)

    assert example_lm.computeProbability(['W']) == -1.746387
    assert example_lm.computeProbability(['K']) == -1.510698
    assert example_lm.computeProbability(['EY']) == -1.708139
    assert example_lm.computeProbability(['AA','DH']) == -1.810896
    assert example_lm.computeProbability(['Z','HH']) == -1.523421
    assert example_lm.computeProbability(['AA','TH']) == (-2.352603 + -2.40901)


def test_language_model_loading_speed():
    print('Start test:',datetime.datetime.now())
    example_lm = languagemodel.LanguageModel(large_language_model_file)
    print('Finish test:',datetime.datetime.now())

test_language_model_loading_speed()
