from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import mxnet as mx


def save_checkpoint(prefix, epoch, arg_params, aux_params):
    """Checkpoint the model data into file.
    :param prefix: Prefix of model name.
    :param epoch: The epoch number of the model.
    :param arg_params: dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    :param aux_params: dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    :return: None
    prefix-epoch.params will be saved for parameters.
    """
    save_dict = {('arg:%s' % k) : v for k, v in list(arg_params.items())}
    save_dict.update({('aux:%s' % k) : v for k, v in list(aux_params.items())})
    param_name = '%s-%04d.params' % (prefix, epoch)
    mx.nd.save(param_name, save_dict)
