import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16Backbone(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()

        # Get pretrained VGG16 from PyTorch's GitHub repo,
        # see https://pytorch.org/hub/pytorch_vision_vgg/
        vgg16 = torch.hub.load('pytorch/vision:v0.5.0', 'vgg16', pretrained=pretrained)

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

    def __init__(self, pretrained_backbone=False):
        super().__init__()
        self._backbone = VGG16Backbone(pretrained=pretrained_backbone)
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

        # Construct new output tensor. If we would not do this and
        # modify the pseudo_predictions inplace in the regularization
        # below, we would cause a RunTimeError from PyTorch during gradient
        # computation.
        output = torch.zeros_like(pseudo_predictions)
        output[:, :5] = pseudo_predictions[:, :5]

        # Regularizing cosθ and sinθ so that the squared sum equals 1.
        # Because then in matches the format of ground truth and we do
        # not need to do this in the loss function.
        scale = torch.sqrt(1. / (torch.pow(pseudo_predictions[:, 5], 2) + torch.pow(pseudo_predictions[:, 6], 2)))
        output[:, 5] = pseudo_predictions[:, 5] * scale  # cosine
        output[:, 6] = pseudo_predictions[:, 6] * scale  # sine

        # We do not apply softmax to tr and tcl pseudo predictions here,
        # because this is done by torch.nn.functional.cross_entropy in
        # the loss function.

        return output


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from dataloader.Eco2018Loader import DeviceLoader, Eco2018
    from utils import softmax

    torch.autograd.set_detect_anomaly(True)

    means = (77.125, 69.661, 65.885)
    stds = (9.664, 8.175, 7.810)
    from augmentation.augmentation import RootAugmentation
    train_transforms = RootAugmentation(mean=means, std=stds)

    model = Textnet(pretrained_backbone=False)
    data_root = '../data/Eco2018-Test'
    train_loader = DeviceLoader(
        DataLoader(Eco2018(data_root=data_root, transformations=train_transforms), shuffle=True, batch_size=1))

    optimizer = torch.optim.Adam(model.parameters(), lr=1)

    for batch, *maps in train_loader:
        model.train()
        output = model(batch)

        # --------------------------------------------------------------------------------
        # Do some sanity checks on the model output
        # --------------------------------------------------------------------------------
        print('Output shape:', output.shape)

        softmaxed_output = softmax(output)

        tr_pred = softmaxed_output[0, :2]
        tcl_pred = softmaxed_output[0, 2:4]
        radii_pred = softmaxed_output[0, 4]
        cos_pred = softmaxed_output[0, 5]
        sin_pred = softmaxed_output[0, 6]

        # Two softmaxed channels of both tr_pred and tcl_pred must sum to 1.
        assert torch.all(torch.isclose(tr_pred[0] + tr_pred[1], torch.tensor(1.)))
        assert torch.all(torch.isclose(tcl_pred[0] + tcl_pred[1], torch.tensor(1.)))

        # Squared sum of each sin_pred[i, j] and cos_pred[i, j] must be 1
        assert torch.all(torch.isclose(cos_pred ** 2 + sin_pred ** 2, torch.tensor(1.)))

        # --------------------------------------------------------------------------------
        # Check if the model is learning, i.e. that the weights receive updates
        # --------------------------------------------------------------------------------
        from loss.loss import loss_fn

        loss = loss_fn(output, maps)
        loss.backward()
        old_weights = [module.weight.data.clone() for module in model.modules() if hasattr(module, 'weight')]
        optimizer.step()
        new_modules = [module for module in model.modules() if hasattr(module, 'weight')]

        grads_missing = False
        grads_zero = False
        for i, m in enumerate(new_modules):
            if m.weight.grad is None:
                grads_missing = True
                print(f'Gradient of module {i} ({m}) is none')

            if not torch.any(m.weight.grad != 0):
                grads_zero = True
                print(f'Gradient of module {i} ({m}) is all 0')

        assert not grads_missing, 'Found weights without gradients'
        assert not grads_zero, 'Found gradients that are all 0'

        total_weights = list(map(lambda x: x.numel(), old_weights))
        total_weights_not_updated = 0
        for i, new_module in enumerate(new_modules):
            weights_not_updated = torch.sum(torch.isclose(old_weights[i], new_module.weight))

            if weights_not_updated > 0:
                #print(f'Module {i} ({new_module}): {weights_not_updated}/{total_weights[i]} weights not updated')
                total_weights_not_updated += torch.sum(weights_not_updated)

        assert total_weights_not_updated < sum(total_weights),\
            f'Nothing of the {sum(total_weights)} weights got updated'

        break
