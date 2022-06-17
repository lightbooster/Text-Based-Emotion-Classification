import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from sklearn.metrics import f1_score, classification_report

from models import load_pretrained_model


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, saving_bert=False):
    print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))

    model_state_dict = model.state_dict().copy()

    if not saving_bert:
        bert_keys = [key for key in model_state_dict.keys() if 'bert' in key]
        for key in bert_keys:
            del model_state_dict[key]

    torch.save({'iteration': iteration,
                'state_dict': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def warm_start(checkpoint_path, model, loading_bert=False, ignore_classifier=False):
    assert os.path.isfile(checkpoint_path)
    print("Loading model state '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    if ignore_classifier:
        for key in checkpoint_dict['state_dict'].keys():
            if 'classifier' in key:
                checkpoint_dict['state_dict'].pop(key)

    model = load_pretrained_model(model, checkpoint_dict['state_dict'], loading_bert)
    print("Loaded model state '{}' with{} BERT".format(checkpoint_path,
                                                       'out' if not loading_bert else ''))

    return model


def load_checkpoint(checkpoint_path, model, optimizer, loading_bert=False):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    model = load_pretrained_model(model, checkpoint_dict['state_dict'], loading_bert)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' with{} BERT from iteration {}" .format(checkpoint_path, 
                                                                         'out' if not loading_bert else '',
                                                                         iteration))

    return model, optimizer, learning_rate, iteration


def validate(model: nn.Module, eval_loader: DataLoader, criterion, device: torch.device):
    f1_micro = f1_macro = f1_weighted = eval_loss = 0.0
    batch_to_device = eval_loader.dataset.batch_to_device

    model.eval()
    print('\nValidating model ...')
    with torch.no_grad():
        outputs_gathered = []
        targets_gathered = []
        for batch in eval_loader:
            inputs, targets = batch_to_device(batch, device=device)

            outputs = model(inputs)

            outputs_detached = outputs.detach().to('cpu')
            if isinstance(criterion, CrossEntropyLoss):
                outputs_detached = torch.softmax(outputs_detached, dim=-1)
                outputs_detached = torch.argmax(outputs_detached,  dim=-1)
                targets_gathered.append(targets.detach().argmax(dim=-1).to('cpu'))
            elif isinstance(criterion, BCEWithLogitsLoss):
                outputs_detached = torch.sigmoid(outputs_detached)
                outputs_detached = torch.round(outputs_detached)
                targets_gathered.append(targets.detach().to('cpu'))
            else:
                raise ValueError("Do not know output aggregation for criterion ", criterion)
            outputs_gathered.append(outputs_detached)

            loss = criterion(outputs, targets)
            eval_loss += loss.item()

        eval_loss = eval_loss / len(eval_loader)
        outputs_gathered = torch.cat(outputs_gathered)
        targets_gathered = torch.cat(targets_gathered)

        f1_micro = f1_score(targets_gathered, outputs_gathered, average='micro')
        f1_macro = f1_score(targets_gathered, outputs_gathered, average='macro')
        f1_weighted = f1_score(targets_gathered, outputs_gathered, average='weighted')
        print(f'Validation results: f1_micro={f1_micro:.3f}|f1_macro={f1_macro:.3f}|'
              f'f1_weighted={f1_weighted:.3f}|loss = {eval_loss:.3f}\n')
        print(classification_report(targets_gathered, outputs_gathered))
        print('\n')

    model.train()

    return (f1_micro, f1_macro, f1_weighted), eval_loss
