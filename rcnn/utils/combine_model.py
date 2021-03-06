from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from builtins import dict
from future import standard_library
standard_library.install_aliases()
from .load_model import load_checkpoint
from .save_model import save_checkpoint


def combine_model(prefix1, epoch1, prefix2, epoch2, prefix_out, epoch_out):
    args1, auxs1 = load_checkpoint(prefix1, epoch1)
    args2, auxs2 = load_checkpoint(prefix2, epoch2)
    arg_names = list(args1.keys()) + list(args2.keys())
    aux_names = list(auxs1.keys()) + list(auxs2.keys())
    args = dict()
    for arg in arg_names:
        if arg in args1:
            args[arg] = args1[arg]
        else:
            args[arg] = args2[arg]
    auxs = dict()
    for aux in aux_names:
        if aux in auxs1:
            auxs[aux] = auxs1[aux]
        else:
            auxs[aux] = auxs2[aux]
    save_checkpoint(prefix_out, epoch_out, args, auxs)
