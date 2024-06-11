import os
import torch
import PIL.Image
import torchvision
import numpy as np

from .models import LinkNet34


class Segmentor:
    def __init__(self, device):
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.device = device
        self.model_path = os.path.join(self.curr_dir, "linknet.pth")
        self.model = LinkNet34()
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model = self.model.to(device)
        self.model.eval()
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def segment(self, image_path, save_path):
        with torch.no_grad():
            image = PIL.Image.open(image_path).convert("RGB")
            image = self.transforms(image)
            image = image.reshape([1, 3, 256, 256])
            image = image.to(self.device)
            pred = self.model(image)
            pred = (pred > 0.8).float()
            pred = pred.detach().cpu().numpy().reshape([256, 256])
            np.save(save_path, pred)
