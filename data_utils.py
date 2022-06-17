
from abc import abstractmethod, abstractstaticmethod
from math import ceil
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Iterator
from omegaconf import DictConfig

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.nn.utils.rnn import pack_sequence, PackedSequence

import fasttext
import fasttext.util
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer

from text_processing import TextProcessingPipeline


class SequenceRandomBatchSampler(Sampler):
    def __init__(self, data_source: Dataset, batch_size: int,
                 shuffle=True, groups_num=5, drop_last=False) -> None:
        self.index_lengths_map = {}
        for i in range(len(data_source)):
            text, _ = data_source[i]
            if isinstance(text, str):
                text = word_tokenize(text)
            self.index_lengths_map[i] = len(text)
        self.index_lengths_map = {key: value for key, value in sorted(self.index_lengths_map.items(), key=lambda item: item[1])}
        self.sorted_indexes = np.array(list(self.index_lengths_map.keys()))

        self.groups  = np.array_split(self.sorted_indexes, groups_num)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __len__(self) -> int:
        return ceil(self.sorted_indexes.size / self.batch_size)
    
    def __iter__(self) -> Iterator:
        if self.shuffle:
            groups = [np.random.permutation(group) for group in self.groups]
            np.random.shuffle(groups)
        else:
            groups = self.groups
        indexes = np.concatenate(groups)
        batches = np.split(indexes, np.arange(self.batch_size, indexes.size, self.batch_size))
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        return iter(batches)


class EmotionsTextDataset(Dataset):
    def __init__(self, path: str, hparams: DictConfig) -> None:
        self.data = pd.read_csv(path, sep='\t')
        self.processing_pipeline = None
        self.classes_num = hparams.classes_num

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def get_target(classes: List[int], classes_num: int) -> torch.Tensor:
        target = np.zeros(classes_num)
        for index in classes:
            target[index] = 1
        return torch.tensor(target)

    def get_text_and_classes(self, index: int):
        text, emotions, _ = self.data.iloc[index]
        if self.processing_pipeline is not None:
            text = self.processing_pipeline(text)
        if isinstance(emotions, str):
            classes = [int(emotion) for emotion in emotions.split(',')]
        else:
            classes = [emotions]
        return text, classes

    @abstractmethod
    def __getitem__(self, index):
        ...

    @abstractstaticmethod
    def batch_to_device(batch, device):
        ...

    def get_class_weights(self, wtype='max'):
        if wtype not in ['max', 'sum']:
            raise ValueError(f"Undefined class weights type '{wtype}'")
        targets = [target for _, target in self]
        positive_samples = torch.stack(targets).sum(axis=0)
        if wtype == 'max':
            return 1.5 - (1 / (max(positive_samples) / positive_samples))
        else:
            return positive_samples.sum() / (self.classes_num * positive_samples)


class EmotionsTextWithContextDataset(EmotionsTextDataset):
    def __init__(self, path: str, hparams: DictConfig) -> None:
        super().__init__(path, hparams)

    def get_text_and_classes(self, index: int):
        text, emotions, _, context = self.data.iloc[index]
        if self.processing_pipeline is not None:
            text = self.processing_pipeline(text)
            context = self.processing_pipeline(context)
        classes = [int(emotion) for emotion in emotions.split(',')]
        return (text, context), classes


class FastTextDataset(EmotionsTextDataset):
    def __init__(self, path: str, hparams: DictConfig) -> None:
        super().__init__(path, hparams)
        self.fasttext_model = fasttext.load_model(hparams.fasttext.checkpoint_path)
        if hparams.fasttext.text_preprocessing:
            self.processing_pipeline = TextProcessingPipeline.get_standard_pipeline()
    
    @staticmethod
    def batch_to_device(batch, device):
        texts, targets = batch
        texts = texts.to(device=device)
        targets = targets.to(device=device)
        return texts, targets

    @staticmethod
    def collate_fn(batch) -> Tuple[PackedSequence, torch.Tensor]:
        texts, targets = zip(*batch)
        texts = pack_sequence(texts, enforce_sorted=False)
        targets = torch.stack(targets)
        return texts, targets

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text, classes = self.get_text_and_classes(index)
        # text
        text = word_tokenize(text)
        text = [self.fasttext_model[word] for word in text]
        text = np.stack(text)
        text = torch.tensor(text)
        # target
        target = self.get_target(classes, self.classes_num)
        return text, target


class BertTextDataset(EmotionsTextDataset):
    def __init__(self, path: str, hparams: DictConfig) -> None:
        super().__init__(path, hparams)
        self.collate_fn = self.BertCollator(hparams.bert.checkpoint_path)
        if hparams.bert.text_preprocessing:
            self.processing_pipeline = TextProcessingPipeline.get_standard_pipeline()

    @staticmethod
    def batch_to_device(batch, device):
        (tokens, lengths), targets = batch
        tokens  = {key: value.to(device=device) for key, value in tokens.items()}
        targets = targets.to(device=device)
        return (tokens, lengths), targets

    class BertCollator:
        def __init__(self, bert_tokenizer_path) -> None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
        
        def __call__(self, batch) -> Tuple[Tuple[Dict[str, torch.Tensor], torch.Tensor], torch.Tensor]:
            texts, targets = zip(*batch)
            tokens = self.bert_tokenizer(texts, return_tensors="pt", padding=True)
            targets = torch.stack(targets)
            lengths = torch.sum(tokens.attention_mask, 1)
            return (tokens, lengths), targets

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        text, classes = self.get_text_and_classes(index)
        # target
        target = self.get_target(classes, self.classes_num)
        return text, target


class BertTextSEPContextDataset(BertTextDataset, EmotionsTextWithContextDataset):
    def __init__(self, path: str, hparams: DictConfig) -> None:
        super().__init__(path, hparams)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        (text, context), classes = self.get_text_and_classes(index)
        if context is not None:
            text = context + " [SEP] " + text
        # target
        target = self.get_target(classes, self.classes_num)
        return text, target


class BertTextWithContextDataset(BertTextDataset, EmotionsTextWithContextDataset):
    def __init__(self, path: str, hparams: DictConfig) -> None:
        super().__init__(path, hparams)
        self.collate_fn = self.BertWithContextCollator(hparams.bert.checkpoint_path)

    @staticmethod
    def batch_to_device(batch, device):
        ((texts_tokens, texts_lengths), (contexts_tokens, contexts_lengths)), targets = batch
        texts_tokens     = {key: value.to(device=device) for key, value in texts_tokens.items()}
        contexts_tokens  = {key: value.to(device=device) for key, value in contexts_tokens.items()}
        targets = targets.to(device=device)
        return ((texts_tokens, texts_lengths), (contexts_tokens, contexts_lengths)), targets

    class BertWithContextCollator:
        def __init__(self, bert_tokenizer_path) -> None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
        
        def __call__(self, batch) -> Tuple[Tuple, torch.Tensor]:
            texts, targets  = zip(*batch)
            texts, contexts = zip(*texts)
            # text
            texts_tokens = self.bert_tokenizer(texts, return_tensors="pt", padding=True)
            texts_lengths = torch.sum(texts_tokens.attention_mask, 1)
            # context
            contexts_tokens = self.bert_tokenizer(contexts, return_tensors="pt", padding=True)
            contexts_lengths = torch.sum(contexts_tokens.attention_mask, 1)
            # target
            targets = torch.stack(targets)
            return ((texts_tokens, texts_lengths), (contexts_tokens, contexts_lengths)), targets

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        text, classes = self.get_text_and_classes(index)
        text, context = text
        if context is None:
            context = "[PAD]"
        # target
        target = self.get_target(classes, self.classes_num)
        return (text, context), target


def prepare_dataloader(model_recipe: DictConfig, hparams, dtype='train'):
    # datasets
    if dtype not in ['train', 'eval', 'test']:
        raise ValueError("Choose dataset type between 'train', 'eval' or 'test'")
    dataset = None
    if model_recipe.word_embedding == 'BERT':
        if model_recipe.use_context and model_recipe.context_type in ['cls-concat', 'emo-concat']:
            dataset_class = BertTextWithContextDataset
        elif model_recipe.use_context and model_recipe.context_type == 'sep':
            dataset_class = BertTextSEPContextDataset
        elif model_recipe.use_context:
            raise ValueError(f"No such context_type named '{model_recipe.context_type}'; "
                              "choose between 'cls-concat' or 'emo-concat'")
        else:
            dataset_class = BertTextDataset
        dataset = dataset_class(hparams[dtype + "_dataset"].path, hparams)
    elif model_recipe.word_embedding == 'FastText':
        if model_recipe.use_context:
            raise ValueError("Context usage is available only with BERT ")
        dataset = FastTextDataset(hparams[dtype + "_dataset"].path, hparams)
    else:
        raise ValueError(f"No such word_embedding in model recipe named '{model_recipe}'")
    # dataloaders
    batch_sampler = SequenceRandomBatchSampler(dataset,
                                                     batch_size=hparams[dtype + "_dataset"].batch_size,
                                                     shuffle=hparams[dtype + "_dataset"].shuffle,
                                                     groups_num=hparams[dtype + "_dataset"].sampler_groups)

    data_loader = DataLoader(dataset,
                             collate_fn=dataset.collate_fn,
                             batch_sampler=batch_sampler,
                             num_workers=hparams[dtype + "_dataset"].num_workers,
                             prefetch_factor=hparams[dtype + "_dataset"].prefetch_factor)

    return data_loader


def prepare_dataloaders(model_recipe: DictConfig, hparams: DictConfig):
    # datasets
    train_loader = prepare_dataloader(model_recipe, hparams, 'train')
    eval_loader  = prepare_dataloader(model_recipe, hparams, 'eval')

    return train_loader, eval_loader, train_loader.dataset.batch_to_device
