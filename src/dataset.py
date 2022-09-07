import numpy as np
import torch.utils.data
import os
import torch
from torchvision import transforms
import random
import gin
from pathlib import Path
import json


SEED_FIXED = 100000
    

@gin.configurable 
def get_mvdcndata(
        ending='.png',
        root_dir=os.environ['DATA_DIR'],
        make_npy_files=False,
        valid_size=0.2,
        batch_size=8,
        random_seed_for_validation=10,
        num_views=12,
        num_workers=0,
        specific_views=None,
        seed=777,
        use_cuda=True,
        ):
    random.seed(seed)
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu  vars
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    
    test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    test_dataset = MultiviewModelDataset(root_dir, 'test',
        ending=ending,
        num_views=num_views, 
        specific_view=specific_views, 
        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers)

    training = MultiviewModelDataset(root_dir, 'train',
        ending=ending, 
        num_views=num_views, 
        specific_view=specific_views, 
        transform=train_transform)

    num_train = len(training)
    indices = list(range(num_train))
    training_idx = indices

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    
    split = int(np.floor(valid_size * num_train))
    random.Random(random_seed_for_validation).shuffle(indices)
    training_idx, valid_idx = indices[split:], indices[:split]
    
    valid_sub = torch.utils.data.Subset(training, valid_idx)
    valid_loader = torch.utils.data.DataLoader(valid_sub,
                       batch_size=batch_size,
                       shuffle=False,
                       num_workers=num_workers,
                       ) 

    training_sub = torch.utils.data.Subset(training, training_idx)

    training_loader = torch.utils.data.DataLoader(training_sub,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   ) 
    
    return training_loader, valid_loader, test_loader


class MultiviewModelDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, ending='.png',
                 num_views=12, shuffle=True, specific_view=None, transform=None):

        self.root_dir = Path(root_dir)
        metadata_file = Path(root_dir) / 'metadata.json'
        with open(str(metadata_file)) as f:
            self.metadata = json.load(f)

        self.samples = self.metadata[split]
        self.classnames = self.metadata['classnames']
        self.split = split

        self.num_views = num_views
        self.specific_view = specific_view

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        classname = sample['classname']
        model = sample['model']
        class_id = self.classnames.index(classname)
        imgs = torch.load(self.root_dir / self.split / f'{model}.npy')
        trans_imgs = []
        for img, view in zip(imgs[self.specific_view], self.specific_view):
            if self.transform:
                img = self.transform(img)
            trans_imgs.append(img)
        data = torch.stack(trans_imgs)
        return idx, data, class_id
