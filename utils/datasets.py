import os
import cv2
import h5py
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
from albumentations import pytorch as albu_pytorch
from sklearn.model_selection import train_test_split
from utils.utils import make_mask
from configs.train_params import RANDOM_SEED
import warnings
warnings.filterwarnings('ignore')


class SteelSegmentationDataset(Dataset):
    def __init__(self, df, data_dir='../data/train_images', phase='train', mean=None, std=None):
        self.df = df
        self.data_dir = data_dir
        self.transforms = get_transforms(phase, mean, std)

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image = cv2.imread(os.path.join(self.data_dir, image_id))

        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask'][0].permute(2, 0, 1)

        return image, mask

    def __len__(self):
        return len(self.df.index)


class SteelClassificationDataset(Dataset):
    def __init__(self, df, data_dir='../data/train_images', phase='train', mean=None, std=None):
        self.df = df
        self.data_dir = data_dir
        self.transforms = get_transforms(phase, mean, std)

    def __getitem__(self, idx):
        fname, target = self.df.iloc[idx].values
        image = cv2.imread(os.path.join(self.data_dir, fname))
        image = self.transforms(image=image)['image']

        return image, float(target)

    def __len__(self):
        return len(self.df)


class SteelHDF5ClassificationDataset(Dataset):
    def __init__(self, path_to_hdf5, phase='train', mean=None, std=None):
        self.path_to_hdf5 = path_to_hdf5
        h = h5py.File(path_to_hdf5, 'r')
        self.images = h['images']
        self.labels = h['labels']
        h.close()

        self.transforms = get_transforms(phase, mean, std)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.labels[idx]
        image = self.transforms(image=image)['image']

        return image, float(target)

    def __len__(self):
        h = h5py.File(self.path_to_hdf5, 'r')
        len_ = len(h)
        h.close()
        return len_


def get_transforms(phase, mean, std, list_transforms=None):
    if not list_transforms:
        list_transforms = []

    if phase == 'train':
        list_transforms.extend(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.RandomBrightnessContrast(p=0.1, brightness_limit=0.1, contrast_limit=0.1)
            ]
        )
    list_transforms.extend(
        [
            albu.Normalize(mean=mean, std=std),
            albu_pytorch.ToTensor()
        ]
    )

    list_transforms = albu.Compose(list_transforms)
    return list_transforms


def data_provider(df, data_dir, phase, dataset_cls, batch_size, stratify_by, n_workers, mean, std):
    train_df, val_df = train_test_split(df, test_size=.2, stratify=df[stratify_by], random_state=RANDOM_SEED)
    df = train_df if phase == 'train' else val_df

    dataset = dataset_cls(df, data_dir, phase, mean, std)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=n_workers,
                            pin_memory=True,
                            shuffle=True
                            )
    return dataloader


