import os
import time, datetime
import tqdm
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from models import compose_model
from data_utils import prepare_dataloaders
from utils import load_checkpoint, validate, warm_start, save_checkpoint

import hydra
from omegaconf import DictConfig


def train(hparams: DictConfig):
    output_dir = os.path.abspath(hparams.trainer.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print('Output directory: ', output_dir)

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
    lr = hparams.trainer.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=hparams.trainer.weight_decay)
    iteration = 0

    # checkpoint
    if hparams.trainer.warm_start and hparams.trainer.continue_from_checkpoint:
        raise ValueError("Cannot 'Warm start' and 'Continue from checkpoint' at the same time. Check config")
    elif hparams.trainer.warm_start:
        model = warm_start(hparams.trainer.checkpoint_path, model=model,
                           loading_bert=hparams.trainer.load_bert_from_common_checkpoint,
                           ignore_classifier=hparams.trainer.warm_start_ignore_classifier)
    elif hparams.trainer.continue_from_checkpoint:
        model, optimizer, lr, iteration = load_checkpoint(hparams.trainer.checkpoint_path,
                                                          model=model, optimizer=optimizer,
                                                          loading_bert=hparams.trainer.load_bert_from_common_checkpoint)
        iteration += 1

    # data
    train_loader, eval_loader, batch_to_device = prepare_dataloaders(hparams.model.recipe, hparams=hparams)

    # logging
    logger = SummaryWriter(flush_secs=30)

    # loss function
    class_weights = None 
    loss_function = BCEWithLogitsLoss if hparams.model.recipe.multi_label else CrossEntropyLoss
    if hparams.trainer.class_weights != 'none':
        class_weights = train_loader.dataset.get_class_weights(wtype=hparams.trainer.class_weights).to(device)
        criterion = loss_function(reduction='none')
    else:
        criterion = loss_function()


    # prestart validation
    _ = validate(model, eval_loader, loss_function(), device)

    # --- TRAIN LOOP ---
    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    for epoch in itertools.count(start=epoch_offset, step=1):
        print("Epoch: {}".format(epoch))
        progress = tqdm.tqdm(train_loader)
        train_loss_accumulated = 0.0
        for batch in progress:
            model.zero_grad()
            
            inputs, targets = batch_to_device(batch, device=device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            if class_weights is not None:
                loss = (loss * class_weights).mean()
            loss.backward()
            optimizer.step()

            progress.set_postfix_str(f'loss={loss.item():.2f}')
            train_loss_accumulated += loss.item()

            # validate
            if (iteration + 1) % hparams.trainer.validate_every_n_iterations == 0 or \
               (iteration + 1) >= hparams.trainer.max_iterations:
                metrics, eval_loss = validate(model, eval_loader, loss_function(), device)
                f1_micro, f1_macro, f1_weighted = metrics
                checkpoint_name = f"{hparams.name}__iter_{iteration}__f1_macro_{(f1_macro):.2f}.ckpt"
                checkpoint_path = os.path.join(output_dir, checkpoint_name)
                save_checkpoint(model, optimizer, lr, iteration, checkpoint_path, saving_bert=hparams.bert.finetune)

                # log
                train_loss_accumulated = train_loss_accumulated / hparams.trainer.validate_every_n_iterations
                logger.add_scalar('Epoch', epoch, iteration)
                logger.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], iteration)
                logger.add_scalar('Loss/train', train_loss_accumulated, iteration)
                logger.add_scalar('Loss/eval', eval_loss, iteration)
                logger.add_scalar('Evaluation/f1_macro', f1_macro, iteration)
                logger.add_scalar('Evaluation/f1_micro', f1_micro, iteration)
                logger.add_scalar('Evaluation/f1_weighted', f1_weighted, iteration)
                train_loss_accumulated = 0.0

            if iteration + 1 >= hparams.trainer.max_iterations:
                print(f'Reached max iterations {hparams.trainer.max_iterations}. Stopping...')
                return

            iteration += 1
                                    

@hydra.main(config_path="configs", config_name="go_train")
def main(hparams: DictConfig):
    train(hparams)


if __name__ == "__main__":
    main()