from tqdm import tqdm
import numpy as np
from utils.callbacks import callbacks
from utils.metrics import smape
import utils.WandbLogger as WandbLogger

wandb.login()

class Trainer():
    def __init__(self, config):
        self.config = config