from torch import nn
import torch


class ContextEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ContextEmbeddingLoss, self).__init__()
        self.margin = margin

    def similarity(self, xi, xj):
        sim_measure = nn.CosineSimilarity()
        return sim_measure(xi, xj)

    def forward(self, anchor, positive, negatives):
        # positive = contrasts[:, 0]
        # negatives = contrasts[:, 1:]
        positive_loss = self.similarity(anchor, positive)
        n_negatives = negatives.size(1)
        negative_loss = 0
        for i in range(n_negatives):
            negative_loss += self.similarity(anchor, negatives[:, i])
        negative_loss /= n_negatives
        loss, _ = torch.max(positive_loss - negative_loss + self.margin, 0)
        return loss


# anchor = nn.Linear(10, 10)(torch.ones(6, 10))
# contrasts = torch.Tensor()
# for i in range(5):
#     contrasts = torch.cat(
#         [contrasts, nn.Linear(10, 10)(torch.ones(6, 10)).unsqueeze(1)], dim=1)
# criterion = EmbeddingLoss()
# print(criterion(anchor, contrasts).backward())
