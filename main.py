from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os 
import shutil
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F



from train import train
from evaluate import evaluate

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', './config/cifar10.py', 'Training configuration.', lock_config=True)
flags.DEFINE_string('logging_dir', './logs', 'The output directory.')
flags.DEFINE_string('img_dir', './images', 'The sample image directory.')
flags.DEFINE_enum('device', 'cuda:2', ['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], 'device to use')
flags.DEFINE_string('restore_dir', None, 'restore checkpoint')
flags.DEFINE_bool('training', True, 'training or testing')
flags.DEFINE_enum('verbose', 'INFO', ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 'logging level')

# flags.mark_flags_as_required(['training', 'device'])


def main(argv):
    
    # set logging directory
    if not os.path.exists(FLAGS.logging_dir):
        print('Creating log directory.')
        os.makedirs(FLAGS.logging_dir)
    else:
        response = input(f"log directory {FLAGS.logging_dir} already exists. Overwrite? (Y/N)")
        if response.upper() == "Y":
            overwrite = True
            if overwrite:
                shutil.rmtree(FLAGS.logging_dir)
                os.makedirs(FLAGS.logging_dir)
    
    # set logging level
    level = getattr(logging, FLAGS.verbose, None)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(FLAGS.logging_dir, "stdout.txt"))
    formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    if FLAGS.training:
        logging.info("Writing log file to {}".format(FLAGS.logging_dir))
        train(FLAGS)
    else:
        evaluate(FLAGS)

if __name__ == '__main__':
    
    app.run(main)