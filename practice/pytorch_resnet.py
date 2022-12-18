import os
import sys
import tempfile
from webbrowser import get
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets

from torch.nn.parallel import DistributedDataParallel as DDP
from models.resnet import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

DATA_PATH = "./data/"
N_EPOCHS = 200
BATCH_SIZE = 256


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def get_dataset_cifar10():
    # datasetの準備
    cifar10_train = datasets.CIFAR10(
        DATA_PATH, train=True, download=True, transform=transforms.ToTensor())

    train_img_stack = torch.stack([img_t for img_t, _ in cifar10_train])
    mean_train = train_img_stack.view(3, -1).mean(dim=1)
    std_train = train_img_stack.view(3, -1).std(dim=1)

    # 学習の変換の設定
    transforms_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean_train, std_train)
        ]
    )

    transforms_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean_train, std_train)
        ]
    )

    cifar10_train_dataset_aug = datasets.CIFAR10(
        DATA_PATH, train=True, download=True, transform=transforms_train)
    cifar10_test_dataset = datasets.CIFAR10(
        DATA_PATH, train=False, download=True, transform=transforms_test)
    
    return cifar10_train_dataset_aug, cifar10_test_dataset

def get_dataloader():
    train_dataset, test_dataset = get_dataset_cifar10()
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, pin_memory=False, 
        num_workers=0, drop_last=False, shuffle=False)

    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, pin_memory=False, 
        num_workers=0, drop_last=False, shuffle=False)
    
    return train_dataloader, test_dataloader

@torch.no_grad()
def eval_model(model, test_dataloader, loss_fn, device):
    model.eval()
    num_correct = 0
    loss_test = 0.0
    for imgs, labels in test_dataloader:
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)
        outputs = model(imgs)
        loss_test += loss_fn(outputs, labels).item()
        num_correct += sum(outputs.argmax(axis=1) == labels).item()
    
    accuracy = num_correct / len(test_dataloader.dataset)
    return accuracy, loss_test

def train_model():
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    train_dataloader, test_dataloader = get_dataloader()
    print(device)

    # create model and move it to GPU with id rank
    model = resnet34().to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=3*0.001)

    for n_epoch in range(N_EPOCHS):      
        print(n_epoch)
        
        loss_train = 0.0
        for imgs, labels in train_dataloader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        
        accuracy_test, loss_test = eval_model(model, test_dataloader, loss_fn, device)
        writer.add_scalar("Loss/train", loss_train, n_epoch)
        writer.add_scalar("Loss/test", loss_test, n_epoch)
        writer.add_scalar("accuracy/train", accuracy_test, n_epoch)

    writer.flush()


if __name__ == "__main__":
   train_model()