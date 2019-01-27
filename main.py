import logging
import shutil
from collections import defaultdict, deque, OrderedDict
from functools import partial

import torch
import torch.optim as optim
import tqdm
from sklearn.model_selection import KFold, train_test_split
from tensorboardX.writer import SummaryWriter
from torch.utils.data import DataLoader
from schedulers import SuperConvergence
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

import utils
from common import config
from datasets import *
from model import PosTagger, AttentionModel, ConvoModel, MultiScaleBlock

logging.basicConfig(level=logging.INFO)
global_step = 0


def get_optimiser(model: PosTagger, lr: float, name: str = None):
    return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


def add_lr_as_to_writer(writer: SummaryWriter, optimiser: optim.Optimizer, step: int):
    for i, group in enumerate(optimiser.param_groups):
        writer.add_scalar("lr_{}".format(i), group['lr'], global_step=step)


def run(experiment_name: Optional[str],
        epochs: int, batch_size: int,
        base_lr: float,
        kfolds: int,
        run_single_split: bool = False):
    logging.info("Configuring ...")
    if experiment_name is not None:
        path = common.MODELS_PATH / experiment_name
        path.mkdir()
    else:
        path = common.MODELS_PATH / "test"
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()

    shutil.copy2(common.MAIN_CONFIG_PATH, path / common.MAIN_CONFIG_PATH.name)
    shutil.copy2("model.py", path / "model.py")

    logging.info("Loading data ...")
    x_data, y_data = load_data()

    logging.info("Training ...")
    if run_single_split:
        return run_train_test_split(x_data, y_data, path, batch_size, epochs, base_lr)
    return run_kfolds(x_data, y_data, path, batch_size, epochs, base_lr, kfolds)


def run_train_test_split(x_data: np.ndarray, y_data: np.ndarray, path: Path, batch_size: int, epochs: int,
                         base_lr: float):
    last_val_metrics = None

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data)

    train_writer = SummaryWriter((path / "train").as_posix())
    valid_writer = SummaryWriter((path / "valid").as_posix())

    train_dataset = DataLoader(
        dataset=TrainDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=common.USE_CUDA,
        num_workers=config['data_workers']
    )
    valid_dataset = DataLoader(
        dataset=ValidDataset(x_valid, y_valid),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=common.USE_CUDA,
        num_workers=config['data_workers']
    )

    model = ConvoModel(config['tagset_size'],
                      hidden_dim_1=config['hidden_dim_1'],
                      hidden_dim_2=config['hidden_dim_2'],
                      num_layers_1=config['num_layers_1'],
                      num_layers_2=config['num_layers_2'])

    optimiser = get_optimiser(model, base_lr, '')
    if common.USE_CUDA:
        model.cuda()

    for epoch in range(epochs):
        logging.info("Epoch: {} / {}".format(epoch + 1, epochs))
        pbar = tqdm.tqdm(total=len(train_dataset))

        train_step(model, optimiser, train_dataset, epochs, pbar, train_writer)
        last_val_metrics = evaluate(model, valid_dataset, valid_writer)

        logging.info("Val stats:\n{}".format(last_val_metrics))

        pbar.close()
    return last_val_metrics


def run_kfolds(x_data: np.ndarray, y_data: np.ndarray, path: Path, batch_size: int, epochs: int, base_lr: float,
               kfolds: int):
    global global_step
    folder = KFold(n_splits=kfolds, random_state=0)
    submission_predictions = []
    valid_predictions = []
    valid_ground_truths = []

    last_val_metrics = None
    x_test = load_test_set()
    for fold, (train_indices, val_indices) in enumerate(folder.split(x_data, y_data)):
        global_step = 0
        logging.info("Fold: {} / {}".format(fold + 1, kfolds))
        x_train, y_train = x_data[train_indices], y_data[train_indices]
        x_valid, y_valid = x_data[val_indices], y_data[val_indices]

        train_writer = SummaryWriter((path / "train_{}".format(fold)).as_posix())
        valid_writer = SummaryWriter((path / "valid_{}".format(fold)).as_posix())

        train_dataset = DataLoader(
            dataset=TrainDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=common.USE_CUDA
        )
        valid_dataset = DataLoader(
            dataset=ValidDataset(x_valid, y_valid),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=common.USE_CUDA
        )

        model = ConvoModel(config['tagset_size'],
                          hidden_dim_1=config['hidden_dim_1'],
                          hidden_dim_2=config['hidden_dim_2'],
                          num_layers_1=config['num_layers_1'],
                          num_layers_2=config['num_layers_2'])
        logging.info("Number of parameters: {}".format(model.get_parameter_number()))

        optimiser = get_optimiser(model, base_lr, '')
        scheduler = CosineAnnealingLR(
            optimiser, int(epochs * len(train_dataset)),
            eta_min=config['min_eta']
        )
        # scheduler = SuperConvergence(
        #     optimizer=optimiser,
        #     total_steps=epochs * len(train_dataset),
        #     max_lr=config['max_lr'],
        #     warmup_steps=int(config['warmup_steps_percentage'] * epochs * len(train_dataset)),
        #     warmup_learning_rate=config['warmup_learning_rate'],
        #     minimal_lr=config['min_eta']
        # )
        if common.USE_CUDA:
            model.cuda()

        best_acc = -np.inf

        for epoch in range(epochs):
            logging.info("Epoch: {} / {}".format(epoch + 1, epochs))
            pbar = tqdm.tqdm(total=len(train_dataset))

            train_step(model, optimiser, train_dataset, scheduler, pbar, train_writer)
            last_val_metrics = evaluate(model, valid_dataset, valid_writer)
            if last_val_metrics['acc'] > best_acc:
                best_acc = last_val_metrics['acc']
                torch.save(model.state_dict(), path / 'model_{}'.format(fold))

            logging.info("Val stats:\n{}".format(last_val_metrics))

            pbar.close()

        test_dataset = DataLoader(
            dataset=TestDataset(x_test), batch_size=batch_size, shuffle=False, pin_memory=common.USE_CUDA,
            num_workers=config['data_workers']
        )
        model.load_state_dict(torch.load(path / 'model_{}'.format(fold)))

        logging.info("Evaluating ...")
        valid_predictions.extend(model.predict(valid_dataset))
        valid_ground_truths.extend(utils.get_ground_truth_from_dataset(valid_dataset))
        submission_predictions.append(model.predict(test_dataset))
        utils.evaluate_single_submission(submission_predictions[-1],
                                         path / '{}_{}.csv'.format(best_acc, utils.get_time_stamp()))
    utils.evaluate_final_submission(submission_predictions,
                                    path / '{}_{}_final.csv'.format(utils.acc(valid_predictions, valid_ground_truths),
                                                                    utils.get_time_stamp()))
    return last_val_metrics


def train_step(model: PosTagger, optimizer: optim.Optimizer,
               train_data: DataLoader, scheduler: _LRScheduler,
               pbar: tqdm.tqdm, train_writer: SummaryWriter):
    global global_step
    model.train()
    limited_deque = partial(deque, maxlen=20)

    running_values = defaultdict(limited_deque)
    for samples, y_trues, lens, masks in train_data:
        scheduler.step(global_step)
        optimizer.zero_grad()
        if common.USE_CUDA:
            samples, y_trues, lens, masks = samples.cuda(), y_trues.cuda(), lens.cuda(), masks.cuda()
        predictions = model(samples, lens, masks)

        acc = model.acc(predictions, y_trues, lens)
        fscore = model.fscore(predictions, y_trues, lens)
        loss = model.loss(predictions, y_trues, lens, masks)

        loss.backward()
        optimizer.step()

        train_writer.add_scalar('loss', loss, global_step)
        train_writer.add_scalar('fscore', fscore, global_step)
        train_writer.add_scalar('acc', acc, global_step)
        train_writer.add_scalar('lr', max(scheduler.get_lr()), global_step=global_step)
        if isinstance(model, ConvoModel) and global_step % 100 == 0:
            for i, layer in enumerate(model.layers):
                if isinstance(layer, MultiScaleBlock):
                    train_writer.add_scalar('gamma_{}'.format(i), layer.modified_softplus(layer.gamma), global_step=global_step)
                    train_writer.add_scalar('beta_{}'.format(i), layer.softplus(layer.key_strength), global_step=global_step)
                    train_writer.add_figure('dilated_weights_{}'.format(i), layer.get_parameter_figure(),
                                            global_step=global_step)

        running_values['fscore'].append(fscore.item())
        running_values['acc'].append(acc.item())
        running_values['loss'].append(loss.item())

        global_step += 1
        pbar.update(1)
        pbar.set_postfix(OrderedDict({
            key: "%.4f" % np.mean(values) for key, values in running_values.items()
        }))


def evaluate(model: PosTagger, data: DataLoader, writer: Optional[SummaryWriter]) -> dict:
    metrics = defaultdict(list)
    model.eval()
    for samples, y_trues, lens, masks in data:
        if common.USE_CUDA:
            samples, y_trues, lens, masks = samples.cuda(), y_trues.cuda(), lens.cuda(), masks.cuda()
        predictions = model(samples, lens, masks)

        acc = model.acc(predictions, y_trues, lens)
        fscore = model.fscore(predictions, y_trues, lens)
        loss = model.loss(predictions, y_trues, lens, masks)

        metrics['acc'].append(acc.item())
        metrics['fscore'].append(fscore.item())
        metrics['loss'].append(loss.item())

    if writer is not None:
        for key, values in metrics.items():
            writer.add_scalar(key, np.mean(values), global_step=global_step)
    metrics = {k: np.mean(v) for k, v in metrics.items()}
    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Output folder")
    parser.add_argument("--epochs", type=int, default=config['epochs'])
    parser.add_argument("--batch_size", type=int, default=config['batch_size'])
    parser.add_argument("--base_lr", type=float, default=config['base_lr'])
    parser.add_argument("--kfolds", type=int, default=config['kfolds'])
    args = parser.parse_args()

    run(args.output, args.epochs, args.batch_size, args.base_lr, args.kfolds)
