import numpy as np
from abc import ABC
from typing import Dict, Any, Tuple, Union
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from transformers import BertModel


class BertWordEmbedding(nn.Module):
    def __init__(self, checkpoint_path, finetune=False) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(checkpoint_path, local_files_only=True)
        self.finetune = finetune
    
    def forward(self, x: Tuple[Dict[str, torch.Tensor], torch.tensor], pooler_output=False) -> PackedSequence:
        x, lengths = x
        x = self.bert(**x)
        x = x.last_hidden_state if not pooler_output else x.pooler_output
        if not self.finetune:
            x = x.detach()
        if not pooler_output:
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        return x


class EmotionEmbedding(nn.Module):
    def __init__(self, input_size: int, hparams: DictConfig, word_embedding_layer=None) -> None:
        super().__init__()
        self.input_size = input_size
        self.word_embedding_layer = word_embedding_layer
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hparams.model.lstm.hidden_size,
                            num_layers=hparams.model.lstm.num_layers,
                            bidirectional=hparams.model.lstm.bidirectional,
                            dropout=hparams.model.lstm.dropout,
                            batch_first=True)

        self.lstm_output_assemble_type = hparams.model.lstm.output_assemble_type
        if self.lstm_output_assemble_type == 'concat':
            self.emotion_embedding_inp_dim = hparams.model.lstm.hidden_size * \
                                             hparams.model.lstm.num_layers  * \
                                             (2 if hparams.model.lstm.bidirectional else 1)
        elif self.lstm_output_assemble_type == 'sum-last':
            self.emotion_embedding_inp_dim = hparams.model.lstm.hidden_size
        elif self.lstm_output_assemble_type == 'sum':
            self.emotion_embedding_inp_dim = hparams.model.lstm.hidden_size * \
                                             (2 if hparams.model.lstm.bidirectional else 1)
        else:
            raise ValueError(f"LSTM output assemble type '{self.lstm_output_assemble_type}' does not exist")

        self.emotion_embedding_size = hparams.model.emotion_embedding_size
        self.emotion_embedding = nn.Linear(self.emotion_embedding_inp_dim,
                                           self.emotion_embedding_size)
        self.dropout = nn.Dropout(hparams.model.emotion_dropout_p)

    def get_embedding_size(self) -> int:
        return self.emotion_embedding_size

    def forward(self, x: Union[PackedSequence, Tuple]) -> torch.Tensor:
        if self.word_embedding_layer is not None:
            x = self.word_embedding_layer(x, pooler_output=False)
        output, (h_n, c_n) = self.lstm(x)
        if self.lstm_output_assemble_type == 'concat':
            x = torch.concat([h_n[i] for i in range(h_n.shape[0])], axis=1)
        elif self.lstm_output_assemble_type == 'sum-last':
            x = torch.sum(h_n, dim=0)
        elif self.lstm_output_assemble_type == 'sum':
            x, _ = pad_packed_sequence(output, batch_first=True)
            x = torch.sum(x, dim=1)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.emotion_embedding(x)
        return x


class EmotionWithContextEmbedding(EmotionEmbedding):
    def __init__(self, input_size: int, hparams: DictConfig, word_embedding_layer=None, context_type='cls-concat') -> None:
        if word_embedding_layer is None or not isinstance(word_embedding_layer, BertWordEmbedding):
            raise ValueError("word_embedding_layer must be 'BertWordEmbedding' instance")
        if context_type not in ['cls-concat', 'emo-concat']:
            raise ValueError(f"No such context_type named '{context_type}' for EmotionWithContextEmbedding model; "
                              "use base EmotionEmbedding for 'sep' or choose between 'cls-concat' or 'emo-concat'")
        super().__init__(input_size, hparams, word_embedding_layer)
        self.context_type = context_type
        self.emotion_embedding_inp_dim += input_size if context_type == 'cls-concat' else 0
        self.emotion_embedding = nn.Linear(self.emotion_embedding_inp_dim,
                                           self.emotion_embedding_size)

    def get_embedding_size(self) -> int:
        if self.context_type == 'emo-concat':
            return self.emotion_embedding_size * 2
        else:
            return self.emotion_embedding_size

    def forward(self, x: Union[PackedSequence, Tuple]) -> torch.Tensor:
        def forward_lstm(x):
            output, (h_n, c_n) = self.lstm(x)
            if self.lstm_output_assemble_type == 'concat':
                x = torch.concat([h_n[i] for i in range(h_n.shape[0])], axis=1)
            elif self.lstm_output_assemble_type == 'sum-last':
                x = torch.sum(h_n, dim=0)
            elif self.lstm_output_assemble_type == 'sum':
                x, _ = pad_packed_sequence(output, batch_first=True)
                x = torch.sum(x, dim=1)
            x = torch.relu(x)
            x = self.dropout(x)
            return x

        pooler_output = (self.context_type == 'cls-concat')

        x, x_context = x
        x = self.word_embedding_layer(x, pooler_output=False)
        with torch.no_grad():
            x_context = self.word_embedding_layer(x_context, pooler_output=pooler_output)

        x = forward_lstm(x)
        if pooler_output:
            x = torch.concat([x, x_context], dim=-1)
            x = self.emotion_embedding(x)
        else:
            x_context = forward_lstm(x_context)
            x = self.emotion_embedding(x)
            x_context = self.emotion_embedding(x_context)
            x_context = x_context.detach()
            x = torch.concat([x, x_context], dim=-1)
        return x

    
class EmotionClassifier(nn.Module):
    def __init__(self, emotion_embedding: EmotionEmbedding, hparams: DictConfig) -> None:
        super().__init__()
        self.emotion_embedding = emotion_embedding
        hidden_sizes = hparams.model.classifier.hidden_sizes
        self.hidden = [
            nn.Linear(hidden_sizes[i - 1] if i - 1 >= 0 else self.emotion_embedding.get_embedding_size(),
                      hidden_sizes[i])
            for i in range(len(hidden_sizes))
        ]
        classifier_input = hidden_sizes[-1] if len(hidden_sizes) else self.emotion_embedding.get_embedding_size()
        self.classifier = nn.Linear(classifier_input, hparams.model.classes_num)
        self.dropout = nn.Dropout(hparams.model.classifier.dropout_p)
        self.freeze_emotion_embedding = hparams.model.recipe.freeze_emotion_embedding

    def to(self, *args, **kwargs):
        new_self = super(EmotionClassifier, self).to(*args, **kwargs)
        for i in range(len(new_self.hidden)):
            new_self.hidden[i] = new_self.hidden[i].to(*args, **kwargs)
        return new_self
    
    def forward(self, x) -> torch.Tensor:
        embedding = self.emotion_embedding(x)
        if self.freeze_emotion_embedding:
            embedding = embedding.detach()
        x = torch.relu(embedding)
        x = self.dropout(x)
        for layer in self.hidden:
            x = layer(x)
            x = torch.relu(x)
            x = self.dropout(x)
        x = self.classifier(x)
        return x

    def predict_by_emotion_embedding(self, embedding):
        self.eval()
        with torch.no_grad():
            x = torch.relu(embedding)
            x = self.dropout(x)
            for layer in self.hidden:
                x = layer(x)
                x = torch.relu(x)
                x = self.dropout(x)
            x = self.classifier(x)
            self.train()
            return x, embedding


    def inferense(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            embedding = self.emotion_embedding(x)
            x = torch.relu(embedding)
            x = self.dropout(x)
            for layer in self.hidden:
                x = layer(x)
                x = torch.relu(x)
                x = self.dropout(x)
            x = self.classifier(x)
            self.train()
            return x, embedding


def compose_model(model_recipe: DictConfig, hparams: DictConfig):
    word_embedding = None
    embedding_size = None
    if model_recipe.word_embedding == "BERT":
        word_embedding = BertWordEmbedding(hparams.bert.checkpoint_path, finetune=hparams.bert.finetune)
        embedding_size = hparams.bert.embedding_size
    elif model_recipe.word_embedding == "FastText":
        embedding_size = hparams.fasttext.embedding_size
    else:
        raise ValueError(f"No such word_embedding in model recipe named '{model_recipe.word_embedding}'")
    if model_recipe.use_context and model_recipe.context_type in ['cls-concat', 'emo-concat']:
        emotion_embedding = EmotionWithContextEmbedding(embedding_size, hparams, word_embedding,
                                                        model_recipe.context_type)
    else:
        emotion_embedding = EmotionEmbedding(embedding_size, hparams, word_embedding)
    classifier = EmotionClassifier(emotion_embedding, hparams)
    return classifier


def load_pretrained_model(model: nn.Module, state_dict, loading_bert=False):
    state_dict = state_dict.copy()
    if not loading_bert:
        bert_keys = [key for key in state_dict.keys() if 'bert' in key]
        for key in bert_keys:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)

    return model