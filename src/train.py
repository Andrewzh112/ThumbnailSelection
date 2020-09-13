from src.data import ThumbnailDataset, get_loaders
from src.loss import ContextEmbeddingLoss
import torch
import logging

logger = logging.getLogger(__name__)


class TrainerConfig:
    epochs = 10
    batch_size = 64
    lr = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    ckpt_path = None
    margin = 0.1
    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, config):
        self.config = config
        self.dataset = ThumbnailDataset()
        self.train_loader = get_loaders(
            self.dataset, batch_size=config.batch_size)
        self.model = model
        self.optimizer = torch.nn.optim.Adam(self.model.parameters())
        self.criterion = ContextEmbeddingLoss(margin=config.margin)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(
            self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):

        def train_step():
            losses = []
            for batch in self.train_loader:
                batch = {k: v.to(self.config.device)for k, v in batch.items()}
                self.optimizer.zero_grad()
                loss = self.model(**batch)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            return losses

        for epoch in range(self.config.epochs):
            losses = train_step()
            logger.info("training loss: %f", sum(losses) / len(losses))
            torch.save(self.model.state_dict(),
                       f'model_weights/thumbnailselector{epoch}')
