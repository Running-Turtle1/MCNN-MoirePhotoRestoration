from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import torch
import os
import logging
import argparse
from utils import MoirePic, weights_init
from net import MoireCNN

device_id = 2
torch.cuda.set_device(device_id)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Demo for showing results",
)
parser.add_argument(
    "-d",
    "--dataset",
    dest="dataset",
    type=str,
    default="../dataset/TIP-2018-clean/trainData",
    help="Path of training dataset",
)
parser.add_argument(
    "-b",
    "--batchsize",
    type=int,
    default=64,
    dest="batchsize",
    help="Set batchsize for training",
)
parser.add_argument(
    "-s",
    "--save",
    type=str,
    default="./model",
    dest="save",
    help="Path for saving the best model",
)
par = parser.parse_args()

if not os.path.exists(par.save):
    os.mkdir(par.save)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def train(model, train_loader, criterion, optimizer, epoch, use_gpu):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            logging.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def val(model, val_loader, epoch, use_gpu):
    model.eval()

    idx, loss_sum = 0, 0.0
    criterion = nn.MSELoss()

    for data, target in val_loader:
        if use_gpu:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        loss_sum += loss.item()
        idx += 1

    if idx > 0:
        loss_sum /= idx

    logging.info("Val Epoch: {} \tLoss: {:.6f}".format(epoch, loss_sum))

    return loss_sum


if __name__ == "__main__":
    root_x = os.path.join(par.dataset, "source")
    root_y = os.path.join(par.dataset, "target")

    train_ds = MoirePic(root_x, root_y, mode='train', val_split=0.1)
    val_ds = MoirePic(root_x, root_y, mode='val', val_split=0.1)

    use_gpu = torch.cuda.is_available()
    train_loader = DataLoader(
        dataset=train_ds,
        shuffle=True,
        batch_size=par.batchsize,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_ds,
        shuffle=False,
        batch_size=par.batchsize,
        num_workers=8,
        pin_memory=True,
    )
    logging.info("loaded dataset successfully!")
    logging.info(f"the number of training set images: {train_ds.__len__()}")

    model = MoireCNN()
    model.apply(weights_init)

    if use_gpu:
        model = model.cuda()
        logging.info("use GPU")
    else:
        print("use CPU")

    criterion = nn.MSELoss()
    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_loss = float('inf')

    logging.info(f"learning rate: {lr}, batch size: {par.batchsize}")

    for epoch in range(50):
        train(model, train_loader, criterion, optimizer, epoch, use_gpu)
        current_loss = val(model, val_loader, epoch, use_gpu)
        scheduler.step(current_loss)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_loss
        }
        torch.save(checkpoint, os.path.join(par.save, 'latest_checkpoint.pth'))

        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), os.path.join(par.save, 'moire_best_weights.pth'))
            logging.info(f"Saved best model at epoch {epoch} with loss {best_loss:.6f}")


        last_loss = current_loss
