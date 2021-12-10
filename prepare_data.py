import torch

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def prepare_dicts(labels):
    labels_set = set(labels)
    label2id, id2label = {}, {}
    for idx, label in enumerate(labels_set):
        label2id[label] = idx
        id2label[idx] = label
    return label2id, id2label


def get_weights(labels_train, num_labels):
    weights = [0] * num_labels
    for label in labels_train:
        weights[label] += 1
    return [1 / (1 + w) for w in weights]


def prepare_dataloader(tokenizer, texts, labels=None, max_len=512,
                       batch_size=32, is_train=False):
    inputs = tokenizer(text=texts,
                       return_tensors='pt',
                       padding=True,
                       truncation=True,
                       max_length=max_len)

    seqs = torch.tensor(inputs['input_ids'])
    masks = torch.tensor(inputs['attention_mask'])

    if labels:
        dataset = TensorDataset(seqs, masks, torch.tensor(labels))
    else:
        dataset = TensorDataset(seqs, masks)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=is_train)


def prepare_train_data(texts, labels, label2id, max_len, batch_size, tokenizer):
    labels_encoded = [label2id[label] for label in labels]
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts,
        labels_encoded,
        test_size=0.2,
        random_state=42,
        stratify=labels_encoded
    )

    weights = get_weights(labels_train, len(label2id))
    train_dataloader = prepare_dataloader(tokenizer, texts_train, labels_train,
                                          max_len=max_len, batch_size=batch_size,
                                          is_train=True)
    val_dataloader = prepare_dataloader(tokenizer, texts_val, labels_val,
                                        max_len=max_len, batch_size=batch_size,
                                        is_train=False)
    return train_dataloader, val_dataloader, weights


def prepare_prediction_data(texts, max_len, batch_size, tokenizer):
    return prepare_dataloader(tokenizer, texts, max_len=max_len,
                              batch_size=batch_size, is_train=False)
