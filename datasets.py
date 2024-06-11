import os
import torch
import PIL.Image
import torchvision
import numpy as np


class ImageLandmarkDataset(torch.utils.data.Dataset):
    def __init__(self, data_image_list):
        super(ImageLandmarkDataset, self).__init__()
        self.images = data_image_list
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):

        image_path = self.images[item]
        image = PIL.Image.open(image_path)
        image = self.transforms(image)

        landmark_path = image_path.replace("_cropped.jpg", "_landmarks.npy")
        landmark = np.load(landmark_path)
        landmark = torch.from_numpy(landmark).reshape([136, ]).float()  # (136, )

        mask_path = image_path.replace("_cropped.jpg", "_mask.npy")
        mask = np.load(mask_path).reshape([1, 256, 256])
        mask = torch.from_numpy(mask).repeat([3, 1, 1]).float()

        return image, landmark, mask


def get_train_val_loaders(data_dir, batch_size, num_workers):
    all_image_list = [os.path.join(data_dir, m) for m in os.listdir(data_dir) if "_cropped.jpg" in m]
    l = len(all_image_list)
    split = int(l * 0.8)
    train_split = all_image_list[:split]
    val_split = all_image_list[split:]
    train_set = ImageLandmarkDataset(train_split)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_set =ImageLandmarkDataset(val_split)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader
