import torch as th
import numpy as np
import torch.nn.functional as F

class ContrastiveLoss(th.nn.Module):
    "Contrastive loss function"
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
            
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = th.mean((1-label) * th.pow(euclidean_distance, 2) +
                                      (label) * th.pow(th.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
        