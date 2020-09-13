class ModelConfig:
    frame_embed_size = 2048
    audio_embed_size = 2048
    text_embed_size = 256
    nlayers = 4
    hidden_size = 512
    common_embed_size = 256
    dropout = 0.2
    nhead = 4
    latent_size = 128

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TrainerConfig:
    epochs = 10
    batch_size = 2
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
