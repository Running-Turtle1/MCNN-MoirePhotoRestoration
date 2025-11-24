import torch.utils.data as data
from torchvision import transforms
from random import sample
from PIL import Image, ImageFile
import os
from tqdm import tqdm


class MoirePic(data.Dataset):
    def __init__(self, rootX, rootY, mode='train', val_split=0.1):
        self.picX = [os.path.join(rootX, img) for img in os.listdir(rootX)]
        self.picY = [os.path.join(rootY, img) for img in os.listdir(rootY)]
        self.picX.sort()
        self.picY.sort()
        
        allpics = list(zip(self.picX, self.picY))
        total_len = len(allpics)

        split_idx = int(total_len * (1 - val_split))

        if mode == 'train':
            self.pics = allpics[:split_idx]
        elif mode == 'val':
            self.pics = allpics[split_idx:]
        elif mode == 'test':
            self.pics = allpics
        else:
            raise ValueError("mode must be 'train', 'val' or 'test'")            
        self.Len = len(self.pics)
        self.mode = mode


    def __getitem__(self, index):
        tf = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
        path_pair = self.pics[index]
        imgX = Image.open(path_pair[0]).convert('RGB')
        imgY = Image.open(path_pair[1]).convert('RGB')
        return tf(imgX), tf(imgY)

    def __len__(self):
        return self.Len


def weights_init(m):
    classname = m.__class__.__name__
    # just for conv layer
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.fill_(0)


# clean data: delete images whose W or H < 260
# after run this code, i found it just clean 8 illegal images
if __name__ == "__main__":
    root = "../dataset/TIP-2018-clean/trainData"
    input_path = os.path.join(root, "source")
    gt_path = os.path.join(root, "target")
    input_imgs = [os.path.join(input_path, img) for img in os.listdir(input_path)]
    gt_imgs = [os.path.join(gt_path, img) for img in os.listdir(gt_path)]
    input_imgs.sort()
    gt_imgs.sort()

    cot = 0
    loop = tqdm(enumerate(input_imgs), total=len(input_imgs), leave=False)
    for idx, img in loop:
        with open(img, "rb") as f:
            Impar = ImageFile.Parser()
            chunk = f.read(2048)
            count = 2048
            while chunk != "":
                Impar.feed(chunk)
                if Impar.image:
                    break
                chunk = f.read(2048)
                count += 2048
            M, N = Impar.image.size[0], Impar.image.size[1]

        if M < 260 or N < 260:
            os.remove(input_imgs[idx])
            os.remove(gt_imgs[idx])
            cot += 1

        loop.set_postfix(unfit_imgs=cot)

    print("Done! Get %d unfit images." % cot)
