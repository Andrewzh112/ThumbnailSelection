from torch import nn
import torch


class ContextEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ContextEmbeddingLoss, self).__init__()
        self.margin = margin

    def similarity(self, xi, xj):
        sim_measure = nn.CosineSimilarity()
        return sim_measure(xi, xj) + 1

    def forward(self, anchor, positive, negatives):
        # positive = contrasts[:, 0]
        # negatives = contrasts[:, 1:]
        positive_loss = self.similarity(anchor, positive)
        n_negatives = negatives.size(1)
        negative_loss = 0
        for i in range(n_negatives):
            negative_loss += self.similarity(anchor, negatives[:, i])
        negative_loss /= n_negatives
        loss, _ = torch.max(negative_loss - positive_loss + self.margin, 0)
        return loss
