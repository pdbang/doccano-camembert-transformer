
import torch
import numpy as np
import pandas as pd

SENTENCES_SIZE = 128
TOKENS_NUMBER = 256



def transform_label(x):
    if x != 'O':
        return x[2:]
    else:
        return x


def truncate(df, size=SENTENCES_SIZE):
    list_rajout = []
    for index, line in df.iterrows():
        tokens, tags = line.tokens, line.ner_tags
        while len(tokens) > size:
            list_rajout.append([line.id_phrase, tokens[:size], tags[:size]])
            tokens, tags = tokens[size:], tags[size:]
        list_rajout.append([line.id_phrase, tokens, tags])

    return pd.DataFrame(list_rajout, columns = ['id_phrase', 'tokens', 'ner_tags'])


def align_labels_with_tokens(labels, word_ids, tag2id):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else tag2id[labels[word_id]]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = tag2id[labels[word_id]]
            # If the label is B-XXX we change it to I-XXX
            #if label % 2 == 1:
                #label += 1
            new_labels.append(label)

    return new_labels


def encode_tags(tags, encodings, tag2id):
    encoded_labels = []
    for i, ner_tags in enumerate(tags):
        encoded_labels.append(align_labels_with_tokens(ner_tags, encodings[i].word_ids), tag2id)

    return encoded_labels

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
