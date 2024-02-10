import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def load_dataset(args):
    '''
    Load custom dataset
    '''
    pass


class YSDataset(Dataset):
    def __init__(self, data, masks, targets, flags, args, simul_random_aug=False, transform=None, mask_transform=None):
        self.data = data
        self.masks = masks
        self.targets = targets
        self.flags = flags
        
        self.simul_random_aug = simul_random_aug
        self.transform = transform
        self.mask_transform = mask_transform
        self.args = args
        
        self.class_map = {
            'Cyst and Tumor': 0,
            'Normal': 1,
            'LMBD': 2
        }
        self.targets = np.array([self.class_map[t] for t in self.targets])
        self.class_map = {v: k for k, v in self.class_map.items()}
            
    def __getitem__(self, index):
        img, mask, label, flag = self.data[index], self.masks[index], self.targets[index], self.flags[index]

        img = Image.fromarray(np.transpose(img, (1,2,0)))
        mask = Image.fromarray(np.transpose(mask, (1,2,0)))
        
        if self.simul_random_aug:
            # simultaneous Resize
            if self.args.resize:
                resize_rate_w = random.choice([1-self.args.max_resize_rate, 1+self.args.max_resize_rate])
                resize_rate_h = random.choice([1-self.args.max_resize_rate, 1+self.args.max_resize_rate])
                resize = transforms.Resize((int(resize_scale_h*img.size[1]),int(resize_scale_w*img.size[0])))
                img = resize(img)
                mask = resize(mask)

            # simultaneous RandomHorizontalFilp
            if self.args.flip:
                if random.random() < 0.5: 
                    img = transforms.functional.hflip(img)
                    mask = transforms.functional.hflip(mask)
                
            # simultaneous Rotate
            if self.args.rotate:
                angle = random.choice([-1,1])
                img = transforms.functional.rotate(img, angle)
                mask = transforms.functional.rotate(mask, angle)
            
            # simultaneous JawRatio
            if self.args.jawratio:
                perspective_rate = random.choice([-self.args.max_jaw_ratio_rate, self.args.max_jaw_ratio_rate])
                X, Y = img.size
                startpoints = [[0, 0], [X, 0], [X, Y], [0, Y]] 
                endpoints = [[0, 0], [X, 0], [int((1+perspective_rate)*X), Y], [int((-perspective_rate)*X), Y]]
                img = transforms.functional.perspective(img, startpoints, endpoints)
                mask = transforms.functional.perspective(mask, startpoints, endpoints)

        img = self.transform(img)
        mask = self.mask_transform(mask)
            
        return img, mask, label, flag

    def __len__(self):
        return len(self.data)

