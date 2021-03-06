import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler
from ignite.handlers.checkpoint import ModelCheckpoint
from torchvision.datasets import CIFAR100
from dataloaders import CocoDetection
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Normalize, ToTensor
from .model.wideresnet import AttentionWideResNet
from .model.retinanet import AttentionRetinaNet
from tqdm import tqdm

from tensorboardX import SummaryWriter
from .utils.utils import Resizer, Augmenter
import argparse
import json

HOME_PREFIX = '/home/se26956/projects/IRP/pytorch-attention-augmented-convolution/'

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="config path")

args = parser.parse_args()


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer

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

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def get_COCO_loaders(batch_size):
    normalize = Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )


    train_transforms = Compose([
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        normalize
    ])

    test_transform = Compose([
        ToTensor(),
        normalize
    ])

    train_dataset = CocoDetection(HOME_PREFIX+'data/coco_detection/train2017',
                                  HOME_PREFIX+'data/coco_detection/annotations/instances_train2017.json',
                                  img_and_target_transform=Resizer(),
                                  transform=train_transforms)

    test_dataset = CocoDetection(HOME_PREFIX+'data/coco_detection/val',
                                 HOME_PREFIX+'data/coco_detection/annotations/instances_val2017.json',
                                 transform=test_transform)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for input, label in train_loader:
        print(input.size())

    return train_loader, test_loader


def run(config):

    if config['model'] == 'AttentionWideResNet':
        train_loader, val_loader = get_data_loaders(config['batch_size'])
        model = AttentionWideResNet(28, 100, 10, (32, 32), 0.0)
    elif config['model'] == 'AttentionRetinaNet':
        train_loader, val_loader = get_COCO_loaders(config['batch_size'])
        model = AttentionRetinaNet(num_classes=80, input_size=(5,3))
    writer = create_summary_writer(model, train_loader, config["tb_logdir"])
    model.cuda()
    
    log_interval = config['log_interval']
    epochs = config['epochs']
    model = nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    scheduler = CosineAnnealingScheduler(optimizer, 'lr', 0.1, 0.001, len(train_loader))
    
    loss_fn = nn.CrossEntropyLoss().cuda()
    
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device='cuda')
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer_saver = ModelCheckpoint(
        config['checkpoint_dir'],
        filename_prefix="model_ckpt",
        save_interval=1000,
        n_saved=10,
        atomic=True,
        save_as_state_dict=True,
        create_dir=True
    )
    trainer.add_event_handler(Events.ITERATION_COMPLETED,
                              trainer_saver,
                              {
                                  "model": model,
                              })
    evaluator = create_supervised_evaluator(model,
                                            metrics={"accuracy": Accuracy(),
                                                     'CE': Loss(loss_fn)},
                                            device="cuda")

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

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
        writer.add_scalar("training/avg_loss", avg_CE, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

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

        writer.add_scalar("valdation/avg_loss", avg_CE, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()
    writer.close()


config_file = args.config
with open(config_file, 'rb') as infile:
    config = json.load(infile)

run(config)


