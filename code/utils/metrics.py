"""
https://www.kaggle.com/code/himanshunayal/tps-sep-2022-using-smape-for-evaluation
"""
import numpy as np
import torch
from torchmetrics import Metric

def smape(A, F):
    tmp = 2 * np.abs(F - A) / (np.abs(A) + np.abs(F))
    len_ = np.count_nonzero(~np.isnan(tmp))
    if len_ == 0 and np.nansum(tmp) == 0: 
        return 100
    return 100 / len_ * np.nansum(tmp)

# TODO change this competition metric for DNN
class SMAPE_score_competition(Metric):
    def __init__(self, compute_on_step=False):
        super().__init__(compute_on_step=compute_on_step)

        self.add_state("valeur", default=torch.tensor(0.0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("cpt", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, gt_bboxes_list: list, pred_bboxes_list: list):

        self.valeur += torch.tensor(smape(gt_bboxes_list, pred_bboxes_list))
        self.cpt += 1

    def compute(self):
        return self.valeur / self.cpt
