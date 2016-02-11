#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import os
import sys

import h5py

sys.path.insert(1,'/home/mwand/projects/Brainstorm/WithWarpCTC/brainstorm/build/lib.linux-x86_64-2.7')
import brainstorm as bs
print('Imported brainstorm from',bs.__file__)
from brainstorm.data_iterators import OneHot, Minibatches
from brainstorm.handlers import PyCudaHandler

bs.global_rnd.set_seed(42)

CUTOFF=10000
MAXEPOCHS=2
BATCH_SIZE=10
assert sys.argv[1] in [ 'CPU', 'GPU' ]

# ---------------------------- Set up Iterators ----------------------------- #

data_dir = os.environ.get('BRAINSTORM_DATA_DIR', '../data')
data_file = os.path.join(data_dir, 'HutterPrize.hdf5')
ds = h5py.File(data_file, 'r')['split']
x_tr, y_tr = ds['training']['default'][:,0:CUTOFF,:], ds['training']['targets'][:,0:CUTOFF,:]
x_va, y_va = ds['validation']['default'][:], ds['validation']['targets'][:]

getter_tr = OneHot(Minibatches(BATCH_SIZE, default=x_tr, targets=y_tr, shuffle=False),
                   {'default': 205})
getter_va = OneHot(Minibatches(BATCH_SIZE, default=x_va, targets=y_va, shuffle=False),
                   {'default': 205})

# ----------------------------- Set up Network ------------------------------ #

network = bs.tools.create_net_from_spec('classification', 205, 205,
                                        'L1000')

# Uncomment next line to use the GPU
if sys.argv[1] == 'GPU':
    network.set_handler(PyCudaHandler())
    print('USE GPU')
network.initialize(bs.initializers.Gaussian(0.01))

# ----------------------------- Set up Trainer ------------------------------ #

trainer = bs.Trainer(bs.training.MomentumStepper(learning_rate=0.01,
                                                 momentum=0.9))
trainer.add_hook(bs.hooks.ProgressBar())
scorers = [bs.scorers.Accuracy(out_name='Output.outputs.predictions')]
trainer.add_hook(bs.hooks.MonitorScores('valid_getter', scorers,
                                        name='validation', interval=3000,
                                        timescale='update'))
# trainer.add_hook(bs.hooks.SaveBestNetwork('validation.total_loss',
#                                           filename='hutter_lstm_best.hdf5',
#                                           name='best weights',
#                                           criterion='min'))
trainer.add_hook(bs.hooks.StopAfterEpoch(MAXEPOCHS))

# -------------------------------- Train ------------------------------------ #

trainer.train(network, getter_tr, valid_getter=getter_va)
print("Best validation set loss:", max(trainer.logs["validation"]["total_loss"]))
