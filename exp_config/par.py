import yaml
from easydict import EasyDict as edict
# sys.path.append('../nise_lib')
from nise_lib.nise_config import cfg as config
import numpy as np
import pprint


def get_edcfg_from_nisecfg(nise_cfg):
    '''

    :param nise_cfg: 2-level class
    :return:
    '''
    new_cfg = nise_cfg.__dict__
    
    for k, v in new_cfg.items():
        new_cfg[k] = v.__dict__
    new_cfg = edict(new_cfg)
    return new_cfg



def update_config(_config, config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            # if k in _config:
            #     if isinstance(v, dict):
            #         _update_dict(_config, k, v)
            #     else:
            _config[k] = v
            # else:
            #     raise ValueError("{} not exist in _config.py".format(k))


pp = pprint.PrettyPrinter(indent = 2)
config_file = 't.yaml'

aya = get_edcfg_from_nisecfg(config)
pp.pprint(aya)
update_config(aya, config_file)
