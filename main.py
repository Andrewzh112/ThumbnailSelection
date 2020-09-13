from src.train import Trainer
from src.model import ThumbnailSelector
from src.config import TrainerConfig, ModelConfig


if __name__ == '__main__':
    mconfig, tconfig = ModelConfig(), TrainerConfig()
    model = ThumbnailSelector(mconfig)
    trainer = Trainer(model, tconfig)
    trainer.train()
