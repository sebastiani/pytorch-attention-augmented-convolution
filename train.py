import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as distributed
from torch.optim import SGD
from torch.nn.parallel import DistributedDataParallel
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Normalize, ToTensor
from model.wideresnet import AttentionWideResNet

from tqdm import tqdm


def get_data_loaders(batch_size):
    normalize = Normalize(mean=[0.49137254, 0.48235294, 0.4466667],
                          std=[0.247058823, 0.24352941, 0.2615686])
    train_transforms = Compose([
        RandomCrop(32),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
    ])

    test_transform = Compose([
         ToTensor(),
         normalize
    ])

    train_dataset = CIFAR100('./data', train=True, download=True, transform=train_transforms)
    val_dataset = CIFAR100('./data', train=False, download=True, transform=test_transform)
    train_sampler = data.distributed.DistributedSampler(train_dataset)

    train_loader = data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, train_sampler


def run(batch_size, epochs, lr, momentum, log_interval):
    train_loader, val_loader, train_sampler = get_data_loaders(batch_size)
    model = AttentionWideResNet(28, 100, 10, 0.0)
    model.cuda()

    model = DistributedDataParallel(model)

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    #scheduler = o
    loss_fn = nn.CrossEntropyLoss().cuda()
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device='cuda')
    evaluator = create_supervised_evaluator(model,
                                            metrics={"accuracy": Accuracy(),
                                                     'CE': Loss(loss_fn)},
                                            device="cuda")

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.EPOCH_STARTED)
    def set_epoch(engine):
        train_sampler.set_epoch(engine.state.epoch)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        train_sampler.set_epoch(iter + 1)

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_CE = metrics['CE']
        tqdm.write(
            "Training Results - Epoch: {} Avg accuracy: {:.2f} Avg loss: {:.2f}".format(engine.state.epoch,
                                                                                        avg_accuracy,
                                                                                        avg_CE)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_CE = metrics['CE']
        tqdm.write(
            "Validation Results - Epoch: {} Avg accuracy {:.2f} Avg loss: {:.2f}".format(engine.state.epoch,
                                                                                         avg_accuracy,
                                                                                         avg_CE)
        )
        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()



torch.distributed.init_process_group("nccl",
                                     )
run(batch_size=256,
        epochs=500,
        lr=0.01,
        momentum=0.9,
        log_interval=200)


