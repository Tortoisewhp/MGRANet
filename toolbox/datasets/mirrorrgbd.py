import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation
class MirrorRGBD(data.Dataset):
    def __init__(self, cfg, mode='trainval', do_aug=True):
        assert mode in ['train', 'val', 'trainval', 'test', 'test_day', 'test_night'], f'{mode} not support.'
        self.mode = mode
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])
        self.root = cfg['root']
        self.n_classes = cfg['n_classes']
        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))
        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])
        self.val_resize = Resize(crop_size)
        self.mode = mode
        self.do_aug = do_aug
        with open(os.path.join(self.root, f'{mode}.txt'), 'r') as f:
            self.infos = f.readlines()
    def __len__(self):
        return len(self.infos)
    def __getitem__(self, index):
        img_path = self.infos[index].strip()
        img = Image.open(os.path.join(self.root, self.mode, 'image', img_path + '.jpg')).convert('RGB')
        depth = Image.open(os.path.join(self.root, self.mode, 'depth', img_path + '.png')).convert('RGB')
        label = Image.open(os.path.join(self.root, self.mode, 'mask_single', img_path + '.png')).convert('L')
        sample = {
            'image': img,
            'depth': depth,
            'label': label
        }
        if self.mode in ['train', 'trainval'] and self.do_aug:
            try:
                sample = self.aug(sample)
            except:
                print('------',img_path)
        sample = self.val_resize(sample)
        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64) / 255.).long()
        sample['label_path'] = img_path.strip().split('/')[-1] + '.png'
        return sample
    @property
    def cmap(self):
        return [
            (0, 0, 0),  # unlabelled
            (255, 255, 255),  # MIRROR
        ]
