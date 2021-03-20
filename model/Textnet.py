import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        # Get pretrained VGG16 from PyTorch's GitHub repo,
        # see https://pytorch.org/hub/pytorch_vision_vgg/
        vgg16 = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)

        # Extract the five convolutional layers from vgg16 as our backbone
        self.stage1 = nn.Sequential(*[vgg16.features[i] for i in range(5)])
        self.stage2 = nn.Sequential(*[vgg16.features[i] for i in range(5, 10)])
        self.stage3 = nn.Sequential(*[vgg16.features[i] for i in range(10, 17)])
        self.stage4 = nn.Sequential(*[vgg16.features[i] for i in range(17, 24)])
        self.stage5 = nn.Sequential(*[vgg16.features[i] for i in range(24, 31)])

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)

        # torch.Size([N, 64, 256, 256])
        # torch.Size([N, 128, 128, 128])
        # torch.Size([N, 256, 64, 64])
        # torch.Size([N, 512, 32, 32])
        # torch.Size([N, 512, 16, 16])
        return f1, f2, f3, f4, f5


class DeconvAndMerge(nn.Module):
    """h_i + f_{5-i} --> h_{i+1};  see eq. (1)"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        combined_channels = in_channels + out_channels
        self._deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self._conv1x1 = nn.Conv2d(in_channels=combined_channels,
                                  out_channels=combined_channels, kernel_size=1)
        self._conv3x3 = nn.Conv2d(in_channels=combined_channels,
                                  out_channels=out_channels, kernel_size=3,
                                  padding=1)

    def forward(self, h, f):
        h = self._deconv(h)
        x = torch.cat([h, f], dim=1)
        x = F.relu(self._conv1x1(x))
        h_next = F.relu(self._conv3x3(x))

        return h_next


class Textnet(nn.Module):

    def __init__(self):
        super().__init__()
        self._backbone = VGG16Backbone()
        # f5=h1 and f4 have 512 channels
        self._up1 = DeconvAndMerge(in_channels=512, out_channels=256)
        # f3 has 256 channels, and h2 has 256 channels
        self._up2 = DeconvAndMerge(in_channels=256, out_channels=128)
        # f2 has 128 channels, and h3 has 128 channels
        self._up3 = DeconvAndMerge(in_channels=128, out_channels=64)
        # f1 has 64 channels, and h4 has 64 channels
        self._up4 = DeconvAndMerge(in_channels=64, out_channels=32)
        self._final_deconv = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self._predict = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                      padding=1),
            nn.Conv2d(in_channels=16, out_channels=7, kernel_size=1)
        )

    def forward(self, batch):
        f1, f2, f3, f4, f5 = self._backbone(batch)
        h1 = f5

        h2 = self._up1(h1, f4)
        h2 = F.relu(h2)

        h3 = self._up2(h2, f3)
        h3 = F.relu(h3)

        h4 = self._up3(h3, f2)
        h4 = F.relu(h4)

        h5 = self._up4(h4, f1)
        h5 = F.relu(h5)

        h5_deconv = self._final_deconv(h5)
        pseudo_predictions = self._predict(h5_deconv)

        output = torch.zeros_like(pseudo_predictions)
        output[:, :2] = F.softmax(pseudo_predictions[:, :2], dim=1)  # text region
        output[:, 2:4] = F.softmax(pseudo_predictions[:, 2:4], dim=1)  # text center line

        # TODO regularizing cosθ and sinθ so that the squared sum equals 1
        #output[:, 4] = None  # radii
        #output[:, 5] = None  # cosine
        #output[:, 6] = None  # sine
        return output


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from dataloader.Eco2018Loader import DeviceLoader, Eco2018

    means = (77.125, 69.661, 65.885)
    stds = (9.664, 8.175, 7.810)
    from augmentation.augmentation import RootAugmentation
    train_transforms = RootAugmentation(mean=means, std=stds)

    model = Textnet()
    data_root = '../data/Eco2018-Test'
    train_loader = DeviceLoader(
        DataLoader(Eco2018(data_root=data_root, transformations=train_transforms), shuffle=True, batch_size=1))

    for batch, *_ in train_loader:
        output = model(batch)
        print(output.shape)
        break
