import torch.nn as nn
import segmentation_models_pytorch as smp


class BaseModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

    def forward(self, x):
        return self.unet(x)
