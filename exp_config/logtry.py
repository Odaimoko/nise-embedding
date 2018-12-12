import logging
from nise_lib.nise_config import nise_cfg
import os
import time
from pathlib import Path


def create_logger(cfg, cfg_name, phase = 'train'):
    root_output_dir = Path(nise_cfg.PATH.LOG_SAVE_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    
    model = '_'.join([
        nise_cfg.TEST.MODE,
        'task',
        str(nise_cfg.TEST.TASK),
        'gt' if nise_cfg.TEST.USE_GT_VALID_BOX else '',
        'propthres',
        str(nise_cfg.ALG.PROP_HUMAN_THRES),
        '_'.join(['RANGE',
                  str(nise_cfg.TEST.FROM),
                  str(nise_cfg.TEST.TO)] if not nise_cfg.TEST.ONLY_TEST else nise_cfg.TEST.ONLY_TEST),
    ])
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    
    final_output_dir = root_output_dir / model / cfg_name
    
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents = True, exist_ok = True)
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename = str(final_log_file),
                        format = head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    
    return logger, str(final_output_dir)


logger, final_output_dir = create_logger(nise_cfg, 'hieei')
logger.info(final_output_dir)
