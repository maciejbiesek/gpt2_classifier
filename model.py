import torch

from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score
from transformers import (GPT2Config, GPT2ForSequenceClassification,
                          AdamW, get_linear_schedule_with_warmup)


class GPT2Classifier:
    def __init__(self, params, num_tokens, label2id, id2label, cuda=True):
        self.label2id, self.id2label = label2id, id2label
        self.device = 'cuda' if cuda else 'cpu'
        self.params = params
        num_labels = len(self.label2id)
        model_config = GPT2Config.from_pretrained('gpt2', num_labels=num_labels)
        self.model = GPT2ForSequenceClassification.from_pretrained(
            'gpt2', config=model_config
        )
        self.model.resize_token_embeddings(num_tokens)
        self.model.config.pad_token_id = self.model.config.eos_token_id

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.params['lr'],
            eps=self.params['eps']
        )


    def train_one_epoch(self, train_dataloader):
        self.model.train()
        pred_labels, true_labels = [], []
        total_loss = 0

        for batch in train_dataloader:
            seqs = batch[0].type(torch.LongTensor).to(self.device)
            masks = batch[1].type(torch.LongTensor).to(self.device)
            labels = batch[2].type(torch.LongTensor).to(self.device)
            true_labels += labels.cpu().numpy().flatten().tolist()

            # forward pass
            outputs = self.model(
                input_ids=seqs,
                attention_mask=masks,
                labels=labels
            )
            loss, logits = outputs[:2]
            total_loss += loss.item()

            # backward pass
            loss.backward()

            # clipping the norm of the gradients to 1.0, to prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update parameters and the learning rate
            self.optimizer.step()
            self.scheduler.step()

            # clear previously calculated gradients
            self.model.zero_grad()

            # prepare list with predicted labels
            logits = logits.detach().cpu().numpy()
            pred_labels += logits.argmax(axis=-1).flatten().tolist()

        avg_epoch_loss = total_loss / len(train_dataloader)
        return true_labels, pred_labels, avg_epoch_loss


    def validate_one_epoch(self, val_dataloader):
        self.model.eval()
        pred_labels, true_labels = [], []
        total_loss = 0

        for batch in val_dataloader:
            seqs = batch[0].type(torch.LongTensor).to(self.device)
            masks = batch[1].type(torch.LongTensor).to(self.device)
            labels = batch[2].type(torch.LongTensor).to(self.device)
            true_labels += labels.cpu().numpy().flatten().tolist()

            # do not compute gradients
            with torch.no_grad():
                outputs = self.model(
                    input_ids=seqs,
                    attention_mask=masks,
                    labels=labels
                )
                loss, logits = outputs[:2]
                total_loss += loss.item()

                # prepare list with predicted labels
                logits = logits.detach().cpu().numpy()
                pred_labels += logits.argmax(axis=-1).flatten().tolist()

        avg_epoch_loss = total_loss / len(val_dataloader)
        return true_labels, pred_labels, avg_epoch_loss


    def predict(self, test_dataloader):
        self.model.eval()
        pred_labels = []

        for batch in test_dataloader:
            seqs = batch[0].type(torch.LongTensor).to(self.device)
            masks = batch[1].type(torch.LongTensor).to(self.device)

            # do not compute gradients
            with torch.no_grad():
                outputs = self.model(
                    input_ids=seqs,
                    attention_mask=masks
                )
                logits = outputs[0]

                # prepare list with predicted labels
                logits = logits.cpu().detach().numpy()
                pred_labels += logits.argmax(axis=-1).flatten().tolist()

        return [self.id2label[x] for x in pred_labels]


    def fit(self, train_dataloader, val_dataloader):
        num_train_steps = len(train_dataloader) * self.params['epochs']
        num_warmup_steps = int(num_train_steps * 0.1)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )
        self.model.to(self.device)

        metrics = defaultdict(list)
        for epoch in tqdm(range(self.params['epochs'])):
            y_train, y_train_pred, train_loss = \
                self.train_one_epoch(train_dataloader)
            metrics['train_loss'].append(train_loss)
            train_acc = accuracy_score(y_train, y_train_pred)
            metrics['train_acc'].append(train_acc)

            y_val, y_val_pred, val_loss = \
                self.validate_one_epoch(val_dataloader)
            metrics['val_loss'].append(val_loss)
            val_acc = accuracy_score(y_val, y_val_pred)
            metrics['val_acc'].append(val_acc)

            print(
                f"Epoch: {epoch + 1}, ",
                f"train_loss: {train_loss:.3f}, ",
                f"train_acc: {train_acc:.3f}, ",
                f"val_loss: {val_loss:.3f}, ",
                f"val_acc: {val_acc:.3f}"
            )

        return metrics
