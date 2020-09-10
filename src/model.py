import torch
from torch import nn
from torchvision import models
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ThumbnailSelection(nn.Module):
    def __init__(self,
                 frame_embed_size,
                 audio_embed_size,
                 text_embed_size,
                 nlayers,
                 hidden_size,
                 common_embed_size=256,
                 dropout=0.2,
                 nhead=4,
                 latent_size=256):
        """[summary]

        Args:
            frame_embed_size (int): [description]
            audio_embed_size (int): [description]
            text_embed_size (int): [description]
            nlayers (int): [description]
            hidden_size (int): [description]
            common_embed_size (int, optional): [description]. Defaults to 256.
            dropout (float, optional): [description]. Defaults to 0.2.
            nhead (int, optional): [description]. Defaults to 4.
            latent_size (int, optional): [description]. Defaults to 256.
        """
        super(ThumbnailSelection, self).__init__()

        # for aggregation
        concat_dim = audio_embed_size + frame_embed_size + text_embed_size*2
        self.gathered_project = nn.Sequential(
            nn.Linear(concat_dim, common_embed_size),
            nn.ReLU(),
            nn.Linear(common_embed_size, latent_size)
        )

        self.audio_fc = nn.Sequential(
            nn.Linear(audio_embed_size, common_embed_size * 2),
            nn.ReLU(),
            nn.Linear(common_embed_size * 2, common_embed_size)
        )
        self.frames_fc = nn.Sequential(
            nn.Linear(frame_embed_size, common_embed_size * 2),
            nn.ReLU(),
            nn.Linear(common_embed_size * 2, common_embed_size)
        )
        self.description_fc = nn.Sequential(
            nn.Linear(text_embed_size, common_embed_size),
            nn.ReLU(),
            nn.Linear(common_embed_size, common_embed_size)
        )
        self.title_fc = nn.Sequential(
            nn.Linear(text_embed_size, common_embed_size),
            nn.ReLU(),
            nn.Linear(common_embed_size, common_embed_size)
        )

        encoder_layers = TransformerEncoderLayer(
            common_embed_size, nhead, hidden_size, dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_projector = nn.Linear(common_embed_size, latent_size)

        # for targets
        self.thumbnail2latent = models.wide_resnet50_2(pretrained=True)
        self.thumbnail2latent.fc = nn.Sequential(
            nn.Linear(
                self.thumbnail_extractor.classifier[0].in_features,
                latent_size * 4
            ),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size)
        )

    def project_modals(self, audio, title,
                       description, frames):
        """[summary]

        Args:
            audio ([type]): [description]
            title ([type]): [description]
            description ([type]): [description]
            frames ([type]): [description]

        Returns:
            [type]: [description]
        """
        audio = self.audio_fc(audio).unsqueeze(1)
        title = self.title_fc(title).unsqueeze(1)
        description = self.description_fc(description).unsqueeze(1)
        frames = self.frames_fc(frames).unsqueeze(1)
        return (audio, title,
                description, frames)

    def forward(self, frames, title, description, audio, thumbnails):
        """[summary]

        Args:
            frames ([type]): [description]
            title ([type]): [description]
            description ([type]): [description]
            audio ([type]): [description]
            thumbnails ([type]): [description]

        Returns:
            [type]: [description]
        """
        # batch_size = frames.size(0)
        # modals = 4

        ###################
        # VIDEO TO LATENT #
        ###################
        # concated_embeddings = torch.cat([frames,
        #                                  title,
        #                                  description,
        #                                  audio], dim=1)
        # anchor = self.gathered_project(concated_embeddings)
        (audio,
         title,
         description,
         frames) = self.project_modals(audio,
                                       title,
                                       description,
                                       frames)
        embeddings = torch.cat([frames,
                                title,
                                description,
                                audio], dim=1)
        embeddings *= torch.tensor(self.common_embed_size**0.5)
        context_embedding = self.transformer_encoder(embeddings)
        # encoded_embedding, _ = torch.max(context_embedding, dim=1)
        # encoded_embedding = context_embedding.view(
        #     batch_size, modals*self.
        # )
        encoded_embedding = torch.mean(context_embedding, dim=1)
        anchor = self.transformer_projector(encoded_embedding)

        #######################
        # THUMBNAIL TO LATENT #
        #######################
        positive_thumbnail = thumbnails[:, 0, :, :]
        positive = self.thumbnail2latent(positive_thumbnail)

        negatives = []
        for i in range(1, thumbnails.size(1)):
            negative_thumbnail = thumbnails[:, i, :, :]
            negative = self.thumbnail2latent(negative_thumbnail)
            negatives.append(negative)

        return anchor, positive, negatives
