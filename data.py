import torch
from random import randint
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image



#%% for NFBS dataset
class NFBSDataset(Dataset):
   
    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transforms.Compose(
                                            [transforms.ToPILImage(),
                                            transforms.RandomAffine(3, translate=(0.02, 0.09)),
                                            transforms.CenterCrop(235),
                                            # transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.5))
                                            ]
                                            ) if not transform else transform

        self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice
      

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if os.path.exists(os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy")):
            image = np.load(os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy"))
        
        else:
            print('Creating NPY file for: ', self.filenames[idx])
            img_name = os.path.join(
                    self.ROOT_DIR, self.filenames[idx], f"sub-{self.filenames[idx]}_ses-NFB3_T1w_brain.nii.gz"
                    )
            # random between 40 and 130
            # print(nib.load(img_name).slicer[:,90:91,:].dataobj.shape)
            img = nib.load(img_name)
            image = img.get_fdata()

            image_mean = np.mean(image)
            image_std = np.std(image)
            img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
            image = np.clip(image, img_range[0], img_range[1])
            image = image / (img_range[1] - img_range[0])
            np.save(
                    os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy"), image.astype(
                            np.float32
                            )
                    )
        
        slice_idx = randint(40, 100) if self.random_slice else 80
        image = image[:, slice_idx:slice_idx + 1, :].reshape(256, 192).astype(np.float32)

        if self.transform:
            image = self.transform(image)

        # sample = {'image': image, "filenames": }
        return image, self.filenames[idx]
#%% for BraTS dataset
class BraTSDataset(Dataset):
   
    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transforms.Compose(
                                            [transforms.ToPILImage(),
                                            transforms.RandomAffine(3, translate=(0.02, 0.09)),
                                            # transforms.CenterCrop(235),
                                            # transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.5))
                                            ]
                                            ) if not transform else transform

        self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if os.path.exists(os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy")):
            image = np.load(os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy"))
        
        else:
            print('Creating NPY file for: ', self.filenames[idx])
            img_name = os.path.join(
                    self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}_flair.nii.gz"
                    )
            # random between 40 and 130
            # print(nib.load(img_name).slicer[:,90:91,:].dataobj.shape)
            img = nib.load(img_name)
            image = img.get_fdata()

            image_mean = np.mean(image)
            image_std = np.std(image)
            img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
            image = np.clip(image, img_range[0], img_range[1])
            image = image / (img_range[1] - img_range[0])
            np.save(
                    os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy"), image.astype(
                            np.float32
                            )
                    )
        
        slice_idx = randint(80, 100) if self.random_slice else 80
        image = image[:, slice_idx:slice_idx + 1, :].reshape(240, 155).astype(np.float32)
        image = image.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        # sample = {'image': image, "filenames": self.filenames[idx]}
        return image, self.filenames[idx]
    
#%% for IXI dataset
def normalise_percentile(volume):
    """
    Normalise the intensity values in each modality by scaling by 99 percentile foreground (nonzero) value.
    """
    for mdl in range(volume.shape[1]):
        v_ = volume[:, mdl, :].reshape(-1)
        v_ = v_[v_ > 0]  # Use only the brain foreground to calculate the quantile
        p_99 = torch.quantile(v_, 0.99)
        volume[:, mdl, :] /= p_99

    return volume

class IXIDataset(Dataset):
   
    def __init__(self, ROOT_DIR, transform=None, img_size=(256, 256), random_slice=False):
        
        self.transform = transforms.Compose(
                                            [transforms.ToPILImage(),
                                            transforms.RandomAffine(3, translate=(0.02, 0.09)),
                                            transforms.CenterCrop(235),
                                            transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.5))
                                            ]
                                            ) if not transform else transform

        self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")

        self.filenames = [f[:-7] for f in self.filenames] # remove .nii.gz
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

        if not os.path.exists(os.path.join(self.ROOT_DIR, 'processed')):
            os.mkdir(os.path.join(self.ROOT_DIR, 'processed'))
            print('Created folder: ', os.path.join(self.ROOT_DIR, 'processed'))
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if os.path.exists(os.path.join(self.ROOT_DIR, 'processed', f"{self.filenames[idx]}.npy")):
        #     image = np.load(os.path.join(self.ROOT_DIR, 'processed', f"{self.filenames[idx]}.npy"))
        
        else:
            print('Creating NPY file for: ', self.filenames[idx])
            img_name = os.path.join(
                    self.ROOT_DIR, f"{self.filenames[idx]}.nii.gz"
                    )
            
            image = torch.from_numpy(nib.load(img_name).get_fdata()) # 256 256 150
            image = normalise_percentile(image)

            print(image.dtype)
            # image_mean = np.mean(image)
            # image_std = np.std(image)
            # img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
            # image = np.clip(image, img_range[0], img_range[1])
            # image = image / (img_range[1] - img_range[0])
            
            print(image.shape)
           
            np.save(
                    os.path.join(self.ROOT_DIR, 'processed', f"{self.filenames[idx]}.npy"), image.numpy().astype(
                            np.float32
                            )
                    )
        
        slice_idx = randint(50, 150) if self.random_slice else 80
        # image = image[:, slice_idx:slice_idx + 1, :].reshape(256, image.shape[-1])
        image = image[:, :, 150//2].reshape(256, 256)
        

        if self.transform:
            image = self.transform(image)

        return image, self.filenames[idx]
#%% cifar10
def load_cifar10(config, train=True):
    print(config.data.data_dir)
    return datasets.CIFAR10(
                            root=config.data.data_dir, train=train, download=True, 
                            transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

def get_dataset(config, train=True):
    if config.data.dataset == 'CIFAR10':
        return load_cifar10(config, train=train)
    elif config.data.dataset == 'IXI':
        return
    else:
        print('Dataset not found')
        return None
    
def cycle(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y
                            
                          
#%%
def get_data_iter(config, shuffle=True, train=True):

    if config.data.dataset == 'CIFAR10':
        dataset = load_cifar10(config, train=train)
    else:
        print('Dataset not found')

    dataset_cycle = cycle(
            DataLoader(
                    dataset,
                    batch_size=config.training.batch_size, shuffle=shuffle,
                    num_workers=0, drop_last=True
                    )
            )

    return dataset_cycle
    

if __name__ == '__main__':
    ROOT_DIR = '/Volumes/USB_DRIVE/IXI-T1'
    dataset = IXIDataset(ROOT_DIR, transform=None, random_slice=True)
    loader = cycle(DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True))
    images = next(loader)
    
    
    print(dataset.__len__())
    print(images.shape)

    images = make_grid(images, nrow=4)
    save_image(images, 'test_IXI.png')

    # ROOT_DIR = './Datasets/NFBS_Dataset'
    # dataset = NFBSDataset(ROOT_DIR, transform=None, img_size=(256, 256), random_slice=True)
    # loader = cycle(DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True))
    # data = next(loader)
    # images = data['image']
    
    # print(dataset.__len__())
    # print(images.shape)

    # make_grid(images, nrow=4)
    # save_image(images, 'test_NFBS.png')

   
    # ROOT_DIR = './Datasets/BraTS2021_Training_Data'
    # dataset = BraTSDataset(ROOT_DIR, transform=None, img_size=(256, 256), random_slice=True)
    # loader = cycle(DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True))
    # data = next(loader)
    # images = data['image']
    
    # print(dataset.__len__())
    # print(images.shape)

    # make_grid(images, nrow=4)
    # save_image(images, 'test_BraTS.png')



