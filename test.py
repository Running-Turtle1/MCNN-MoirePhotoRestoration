import numpy as np
import os
import math
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from net import MoireCNN
from utils import MoirePic

parser = argparse.ArgumentParser(description='Test Moire Removal')
parser.add_argument('-d', '--dataset', type=str, default='../dataset/TIP-2018-clean/testData', help='Path of test dataset')
parser.add_argument('-m', '--model', type=str, default='./model/moire_best_weights.pth', help='Path of model weights')
parser.add_argument('-b', '--batchsize', type=int, default=4, help='Batch size')
parser.add_argument('--gpu', type=int, default=2, help='GPU ID')
args = parser.parse_args()

def psnr(img1, img2):
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * math.log10(1 / mse)


if __name__ == '__main__':
    device = torch.device(f'cuda:{args.gpu}')
    dataset = MoirePic(os.path.join(args.dataset, 'source'),
                       os.path.join(args.dataset, 'target'),
                       mode = 'test', # don't worry, in order to get all images
                       val_split=0.0
    )
    test_loader = DataLoader(
        dataset=dataset, 
        batch_size=args.batchsize, 
        drop_last=False, 
        num_workers=4, 
        pin_memory=True
    )
    model = MoireCNN().to(device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {args.model}")
    model.eval()

    psnr_all, count = 0.0, 0
    loop = tqdm(test_loader, leave=True)
    with torch.no_grad():
        for data, target in loop:
            data = data.to(device)
            target = target.numpy()

            output = model(data)
            output = output.cpu().numpy()
            
            for i in range(target.shape[0]):
                # single image psnr
                current_psnr = psnr(output[i], target[i])
                psnr_all += current_psnr
            
            count += target.shape[0]
            loop.set_postfix(avg_psnr=f"{psnr_all / count:.4f}")

    print(f'Final Testing Dataset PSNR: {psnr_all / count:.4f} dB')
