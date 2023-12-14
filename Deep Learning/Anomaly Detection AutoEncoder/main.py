from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale
import torch.nn.functional as F
from torch.optim import Adam
import torch
from trainer import train
from model import AE
from datetime import datetime

def create_datagen(data_dir, batch_size = 8):
    transforms = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize([0.5], [0.5]),
    ])
    dataset = ImageFolder(data_dir, transform=transforms)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    return data_loader

if __name__ == '__main__':
    optimizer = Adam
    loss_func = F.mse_loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = create_datagen('data/train')
    val_loader = create_datagen('data/val')

    model = AE(1)
    train(model, optimizer, loss_func, train_loader, val_loader,
          log_dir=f'runs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
          device=device, epochs=100, log_interval=10, save_graph=True)

