import os
import time, datetime
import tqdm
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from models import compose_model
from data_utils import prepare_dataloader
from utils import load_checkpoint, validate, warm_start, save_checkpoint

import hydra
from omegaconf import DictConfig


def validate_model(hparams: DictConfig):
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)

    device = 'cpu'
    if hparams.cuda >= 0:
        device = 'cuda:' + str(hparams.cuda) 
    device = torch.device(device)
    print('Device: ', device)

    # model
    model = compose_model(hparams.model.recipe, hparams=hparams).to(device=device)

    # checkpoint
    model = load_checkpoint(hparams.checkpoint_path,
                            model=model, optimizer=None,
                            loading_bert=hparams.load_bert_from_common_checkpoint)[0]

    # data
    test_loader = prepare_dataloader(hparams.model.recipe, hparams=hparams, dtype='test')

    # loss function
    criterion = BCEWithLogitsLoss() if hparams.model.recipe.multi_label else CrossEntropyLoss()

    # prestart validation
    _ = validate(model, test_loader, criterion, device)
                            

@hydra.main(config_path="configs", config_name="bert_go_emotions_test")
def main(hparams: DictConfig):
    validate_model(hparams)


if __name__ == "__main__":
    main()