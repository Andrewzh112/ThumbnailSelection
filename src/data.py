from torch.utils.data import Dataset, DataLoader
import json
import torch
import numpy as np
from PIL import Image


class ThumbnailDataset(Dataset):
    def __init__(self, n_negatives=5):
        super(ThumbnailDataset, self).__init__()
        self.description = json.load(open('features/description.json'))
        self.audio = json.load(open('features/audio.json'))
        self.title = json.load(open('features/title.json'))
        self.videos = list(self.audio.keys())
        self.frames = [np.load(f'features/{video}.npy')
                       for video in self.videos]
        self.idx2name = dict(enumerate(self.videos))
        self.n_negatives = n_negatives

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        video = self.idx2name[item]
        thumbnail = Image.open(f'data/{video}.jpg')
        negatives = [Image.open(
            f'data/{video}N{i}.jpg') for i in range(self.n_negatives)]
        return {
            'audio': torch.tensor(self.audio[video]),
            'description': torch.tensor(self.description[video]),
            'title': torch.tensor(self.title[video]),
            'frames': torch.tensor(self.frames[item]),
            'thumbnail': torch.tensor(np.asarray(thumbnail)),
            'negatives': torch.cat([torch.tensor(
                np.asarray(
                    negative)).unsqueeze(0) for negative in negatives], dim=0)
        }


def get_loaders(dataset, batch_size=1):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )


if __name__ == '__main__':
    loaders = get_loaders()
    print(next(iter(loaders)))
