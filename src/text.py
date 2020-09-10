from transformers import ElectraTokenizer, ElectraModel
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import numpy as np
import json


def tokenize_text(text):
    tokenizer = ElectraTokenizer.from_pretrained(
        'google/electra-base-discriminator'
    )

    encoded_data_train = tokenizer.batch_encode_plus(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=False,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_data_train['input_ids']
    attention_mask = encoded_data_train['attention_mask']
    token_type_ids = encoded_data_train['token_type_ids']
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }


def get_loader(text, batch_size=8):
    tokenized_text = tokenize_text(text)
    dataset_train = TensorDataset(**tokenized_text)
    return DataLoader(
        dataset_train,
        batch_size=batch_size
    )


class Electra(nn.Module):
    def __init__(self):
        super(Electra, self).__init__()
        self.electra = ElectraModel.from_pretrained(
            'google/electra-small-discriminator',
        )

    def forward(self, inputs):
        return self.electra(**inputs)


def main():
    description_data = json.load(open('data/description.json', 'r'))
    title_data = json.load(open('data/description.json', 'r'))
    description = {}
    title = {}
    for video, text in description_data.items():
        inputs = tokenize_text([text])
        model = Electra()
        output = model(inputs)
        emb = np.mean(output[0].squeeze(0).detach().numpy(), axis=0)
        description[video] = emb.tolist()
    json.dump(description, open('features/description.json', 'w'))
    for video, text in title_data.items():
        inputs = tokenize_text([text])
        model = Electra()
        output = model(inputs)
        emb = np.mean(output[0].squeeze(0).detach().numpy(), axis=0)
        title[video] = emb.tolist()
    json.dump(title, open('features/title.json', 'w'))


if __name__ == '__main__':
    main()
