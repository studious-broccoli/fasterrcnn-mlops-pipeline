import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


class DummyDataset(Dataset):
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        self.transform = T.ToTensor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a dummy image and a dummy target
        image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        image_tensor = self.transform(image)

        target = {
            "boxes": torch.tensor([[10, 10, 100, 100]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64)
        }

        return image_tensor, target

    def collate_fn(self, batch):
        images, targets = list(zip(*batch))
        return list(images), list(targets)


class FasterRCNNLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.005)

    def train_dataloader(self):
        dataset = DummyDataset()
        return DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
