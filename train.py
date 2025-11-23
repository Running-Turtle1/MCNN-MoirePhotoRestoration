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

    # loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        # After Pytorch 0.4, don't need this step 
        # data, target = Variable(data), Variable(target)

        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # loop.set_description(f'Train Epoch [{epoch}/50]')
        # loop.set_postfix(loss = loss.item())
        if batch_idx % 100 == 0:
            logging.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data,
                )
            )
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.data))


def val(model, val_loader, epoch, use_gpu):
    model.eval()

    idx, loss_sum = 0, 0.0
    criterion = nn.MSELoss()

    for data, target in val_loader:
        if use_gpu:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        # data, target = Variable(data), Variable(target)

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
    dataset = MoirePic(
        os.path.join(par.dataset, "source"), os.path.join(par.dataset, "target")
    )
    valdataset = MoirePic(
        os.path.join(par.dataset, "source"), os.path.join(par.dataset, "target"), False
    )

    use_gpu = torch.cuda.is_available()
    train_loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=par.batchsize,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=valdataset,
        shuffle=True,
        batch_size=par.batchsize,
        num_workers=8,
        pin_memory=True,
    )
    logging.info("loaded dataset successfully!")
    logging.info(f"the number of training set images: {dataset.__len__()}")

    model = MoireCNN()
    model.apply(weights_init)
    # model = torch.load("moire_best.pth")

    if use_gpu:
        model = model.cuda()
        # model = nn.DataParallel(model)
        logging.info("use GPU")
    else:
        print("use CPU")

    criterion = nn.MSELoss()
    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)

    best_loss, last_loss = 100.0, 100.0

    logging.info(f"learning rate: {lr}, batch size: {par.batchsize}")

    for epoch in range(50):
        train(model, train_loader, criterion, optimizer, epoch, use_gpu)
        current_loss = val(model, val_loader, epoch, use_gpu)

        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model, os.path.join(par.save, "moire_best.pth"))

        if current_loss > last_loss:
            lr *= 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logging.info(f"Decay learning rate to: {lr}")

        last_loss = current_loss
