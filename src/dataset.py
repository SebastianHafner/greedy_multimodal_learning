import numpy as np
import torch.utils.data
import os
import torch
from torchvision import transforms
import random
import gin
from pathlib import Path
import json
import tifffile
import src.augmentation as aug


SEED_FIXED = 100000
    

@gin.configurable 
def get_urbanmappingdata(
        root_dir,
        train_sites: list = None,
        val_sites: list = None,
        test_sites: list = None,
        sar_bands: list = ('VV', 'VH'),
        opt_bands: list = ('B2', 'B3', 'B4', 'B8'),
        include_unlabeled: bool = False,
        unlabeled_sites: list = None,
        batch_size: int = 8,
        num_workers: int = 0,
        seed: int = 7,
        use_cuda: bool = True):

    random.seed(seed)
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu  vars
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    
    evaluation_transform = transforms.Compose([
            aug.Numpy2Torch()
        ])

    train_transform = transforms.Compose([
            aug.ImageCrop(256),
            aug.RandomFlip(),
            aug.RandomRotate(),
            aug.Numpy2Torch(),
        ])

    train_dataset = UrbanExtractionDataset(root_dir, 'train', train_sites, sar_bands, opt_bands, include_unlabeled,
                                           unlabeled_sites, train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)

    val_dataset = UrbanExtractionDataset(root_dir, 'val', val_sites, sar_bands, opt_bands,
                                         transform=evaluation_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset = UrbanExtractionDataset(root_dir, 'test', test_sites, sar_bands, opt_bands,
                                          transform=evaluation_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                              num_workers=num_workers)

    return train_loader, val_loader, test_loader


# dataset for urban extraction with building footprints
class UrbanExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, split: str, sites: list, sar_bands: list, opt_bands: list,
                 include_unlabeled: bool = False, unlabeled_sites: list = None, transform=None):

        self.root_dir = Path(root_dir)

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.sar_indices = self._get_indices(['VV', 'VH'], sar_bands)
        self.opt_indices = self._get_indices(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'], opt_bands)

        self.split = split
        self.sites = sites if not include_unlabeled else sites + unlabeled_sites
        self.transform = transform

        self.samples = []
        for site in self.sites:
            samples_file = self.root_dir / site / 'samples.json'
            with open(str(samples_file)) as f:
                metadata = json.load(f)
            samples = metadata['samples']
            for sample in samples:
                    sample['labeled'] = True if site in sites else False
            self.samples.extend(samples)

    def __getitem__(self, index):

        sample = self.samples[index]
        site = 'spacenet7' if self.split == 'test' else sample['site']
        patch_id = sample['aoi_id'] if self.split == 'test' else sample['patch_id']
        labeled = sample['labeled']

        img_sar = self._get_sentinel1_data(site, patch_id)
        img_optical = self._get_sentinel2_data(site, patch_id)
        label = self._get_label_data(site, patch_id) if labeled else np.zeros((img_sar.shape[0], img_sar.shape[1]))

        x_sar, x_opt, label = self.transform((img_sar, img_optical, label))

        return index, (x_sar, x_opt), label, labeled

    def _get_sentinel1_data(self, site, patch_id):
        if self.split == 'test':
            file = self.root_dir / site / 'sentinel1' / f'sentinel1_{patch_id}.tif'
        else:
            file = self.root_dir / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
        img = tifffile.imread(file)
        img = img[:, :, self.sar_indices]
        return np.nan_to_num(img).astype(np.float32)

    def _get_sentinel2_data(self, site, patch_id):
        if self.split == 'test':
            file = self.root_dir / site / 'sentinel2' / f'sentinel2_{patch_id}.tif'
        else:
            file = self.root_dir / site / 'sentinel2' / f'sentinel2_{site}_{patch_id}.tif'
        img = tifffile.imread(file)
        img = img[:, :, self.opt_indices]
        return np.nan_to_num(img).astype(np.float32)

    def _get_label_data(self, site, patch_id):
        if self.split == 'test':
            label_file = self.root_dir / site / 'buildings' / f'buildings_{patch_id}.tif'
        else:
            label_file = self.root_dir / site / 'buildings' / f'buildings_{site}_{patch_id}.tif'
        img = tifffile.imread(label_file)
        img = img > 0
        return np.nan_to_num(img).astype(np.float32)

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.sites)} sites.'