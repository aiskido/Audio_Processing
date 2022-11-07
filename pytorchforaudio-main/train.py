import argparse
import os
import numpy as np
from tqdm import tqdm

from utils import config_reader
from utils import train_elements
from utils.plot import plot_history
import torch
from torch.utils.data import DataLoader
from utils.custom_dataset import CustomDataset


def print_metrics(history):
    print(f'train_loss: {history["train_loss"][-1]}\n'
          f'val_loss: {history["val_loss"][-1]}\n'
          f'val_accuracy: {history["val_accuracy"][-1]}\n'
          f'lr: {history["lr"][-1]}')


def train(model, optimizer, loss_fn, scheduler, config_train, dataloaders, device, dst_model):
    print(f'TRAINING ON: {device}')
    model = model
    optimizer = optimizer
    loss_function = loss_fn
    scheduler = scheduler

    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']

    num_epochs = range(config_train['num_epochs'])

    history = {'train_loss': [], 'lr': [],
               'val_loss':[], 'val_accuracy': []
               }

    best_metric = 0
    best_epoch = 0

    for epoch in num_epochs:
        print(f'EPOCH: {epoch}')
        loss_epoch = []
        # print(f'Starting epoch {epoch + 1}')
        model.train()
        for data in tqdm(train_loader):
            # Get inputs
            inputs_train, targets_train = data
            inputs_train, targets_train = inputs_train.to(device), targets_train.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            outputs = model(inputs_train)
            # Compute loss
            loss = loss_function(outputs, targets_train)
            # Perform backward pass
            loss.backward()
            # grads clip
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
            # Perform optimization
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            history['lr'].append(lr)
            loss_epoch.append(loss.data.item())
            # Print statistics
            if scheduler == "StepLR" or scheduler == 'MultiStepLR':
                scheduler.step()

        history['train_loss'].append(np.mean(loss_epoch))

        # validation
        metrics = dict()
        if epoch % config_train['validate_every_epoch'] == 0:
            val_loss_epoch = []
            val_accuracies_epoch = []
            model.eval()
            with torch.no_grad():
                for val_data in tqdm(dev_loader, desc='validation'):
                    inputs_val, targets_val = val_data
                    inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
                    outputs_val = model(inputs_val)
                    true_preds = outputs_val.argmax(dim=1)
                    corrects = (true_preds == targets_val)
                    accuracy = corrects.sum().float() / float(targets_val.size(0))
                    val_accuracies_epoch.append(accuracy.item())
                    val_loss = loss_function(outputs_val, targets_val)
                    val_loss_epoch.append(val_loss.data.item())
            history['val_loss'].append(np.mean(val_loss_epoch))
            # print(val_accuracies_epoch)
            history['val_accuracy'].append(np.mean(val_accuracies_epoch))
        plot_history(history=history, best_epoch=best_epoch, plot_path='', title_key=str(dst_model))
        print_metrics(history)
        if history['val_accuracy'][-1] > best_metric:
            best_epoch = epoch
            best_metric = history['val_accuracy'][-1]
            # save the model below
            torch.save(model.state_dict(), dst_model)



        # log = f"loss: {np.mean(loss_epoch):.3f}, epoch: {epoch}" \
        #       f" acc_train: {metrics['train']['accuracy']:.3f} , acc_dev: {metrics['dev']['accuracy']:.3f} " \
        #       f" best epoch: {best_epoch} , best metric: {best_metric}"


    best = {
        'epoch': best_epoch,
        'metric': best_metric
    }


def main(audio_src, annotations_src, val_src, config_path, dst_model, logs):
    # read configs
    configs = config_reader.read_conf(config_path)
    config_train, config_inference, config_features = configs['training'], configs['inference'], configs['features']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # prepare datasets
    dataset_train = CustomDataset(val_src, audio_src, config_features, device)
    dataset_dev = CustomDataset(val_src, audio_src, config_features, device)
    # prepare dataloaders
    train_loader = DataLoader(dataset=dataset_train, batch_size=config_train['batch'], shuffle=True)
    dev_loader = DataLoader(dataset=dataset_dev, batch_size=config_train['batch'], shuffle=False)
    dataloaders = {'train': train_loader, 'dev': dev_loader}
    # get model
    model, optimizer, loss, scheduler = train_elements.get_train_elements(config_train, device)
    print(f'{optimizer}\n{loss}\n{scheduler}')
    train(model, optimizer, loss, scheduler, config_train, dataloaders, device, dst_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A template for speech projects',
                                     usage='python3 train_.py audio/ labels/ scp/data1/train.scp scp/data1/dev.scp '
                                           'scp/test.scp configs/exp1.yaml models/exp1.pkl log/data1/exp1/')
    parser.add_argument("audio_src", type=str, help="[input] path to a folder with audio files")
    parser.add_argument("annotations_src", type=str, help="[input] path to a folder with annotation files train")
    parser.add_argument("val_annotation_src", type=str, help="[input] path to a folder with annotation files dev")
    parser.add_argument("config", type=str, help="[input] path to a yaml file with configs")
    parser.add_argument("dst_model", type=str, help="[output] path to save a model")
    parser.add_argument("logs", type=str, help="[output] path to a directory with logs")
    args = parser.parse_args()
    audio_src = args.audio_src
    annotations_src = args.annotations_src
    val_src = args.val_annotation_src
    config = args.config
    dst_model = args.dst_model
    logs = args.logs

    main(audio_src, annotations_src, val_src, config, dst_model, logs)
