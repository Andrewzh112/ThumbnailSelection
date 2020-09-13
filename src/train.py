from src.data import ThumbnailDataset, get_loaders
from src.loss import ContextEmbeddingLoss
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, config):
        self.config = config
        self.dataset = ThumbnailDataset()
        self.train_loader = get_loaders(
            self.dataset, batch_size=config.batch_size)
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = ContextEmbeddingLoss(margin=config.margin)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def train(self):
        def train_step():
            losses = []
            for batch in self.train_loader:
                batch['thumbnails'] = batch['thumbnails'].type(
                    torch.FloatTensor)
                batch = {k: v.to(self.device)for k, v in batch.items()}
                self.optimizer.zero_grad()
                anchor, positive, negatives = self.model(**batch)
                loss = self.criterion(anchor, positive, negatives)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            return losses
        pbar = tqdm(range(self.config.epochs))
        for epoch in pbar:
            losses = train_step()
            epoch_loss = sum(losses) / len(losses)
            torch.save(self.model.state_dict(),
                       f'weights/thumbnailselector{epoch}')
            pbar.set_postfix({'train_loss': '{:.3f}'.format(epoch_loss)})
