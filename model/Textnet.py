"""
Components of the FCN proposed by Long et al., which we termed Textnet. See fig.
4 of the paper ("Network architecture").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16Backbone(nn.Module):
    """
    The five convolutional and max-pooling layers of a VGG-16, serving as
    backbone network for Textnet. These are the blue boxes in fig. 4 of the
    paper ("Network architecture").
    """

    def __init__(self, pretrained=False):
        """
        :param pretrained: whether to load the VGG-16 pretrained in ImageNet
        """
        super().__init__()

        # Get (pretrained) VGG16 from PyTorch's GitHub repo,
        # see https://pytorch.org/hub/pytorch_vision_vgg/
        vgg16 = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=pretrained)

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
    """
    Up-sampling layer. Combines a deconvolutional with two convolutional layers.
    Corresponds to a pair of green and yellow boxes in fig. 4 of the paper
    ("Network architecture")).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        combined_channels = in_channels + out_channels
        self._deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1))

        self._conv1x1 = nn.Conv2d(
            in_channels=combined_channels,
            out_channels=combined_channels,
            kernel_size=(1, 1))

        self._conv3x3 = nn.Conv2d(
            in_channels=combined_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=(1, 1))

    def forward(self, h, f):
        """
        h_i + f_{6-i} --> h_{i+1};  see eq. (1) and (2) of the paper
        """
        h = self._deconv(h)
        x = torch.cat([h, f], dim=1)
        x = F.relu(self._conv1x1(x))
        h_next = F.relu(self._conv3x3(x))

        return h_next


class Textnet(nn.Module):
    """
    Textnet combines the backbone VGG-16 with five up-sampling layers (instances
    of DeconvAndMerge, the feature merging network). The i-th up-sampling layer
    receives as input the (i-1)-th up-sampling layer's output (h_{i-1}), and the
    output of the (6-i)-th backbone convolutional layer (f_{6-i}). See fig. 4 of
    the paper ("Network architecture"). Final output P has dimensions Nx7xHxW
    (N: batchsize, H: input image height, W: input image width).
    """

    def __init__(self, pretrained_backbone=False):
        super().__init__()
        self._backbone = VGG16Backbone(pretrained=pretrained_backbone)
        # f5=h1 and f4 have 512 channels
        self._up1 = DeconvAndMerge(in_channels=512, out_channels=256)
        # f3 and h2 have 256 channels
        self._up2 = DeconvAndMerge(in_channels=256, out_channels=128)
        # f2 and h3 have 128 channels
        self._up3 = DeconvAndMerge(in_channels=128, out_channels=64)
        # f1 and h4 have 64 channels
        self._up4 = DeconvAndMerge(in_channels=64, out_channels=32)
        self._final_deconv = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1))
        self._predict = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(in_channels=16, out_channels=7, kernel_size=(1, 1))
        )

    def forward(self, batch):
        """
        Returns tensor with shape Nx7xHxW, consisting of
            1, 2) logits for text region
            3, 4) logits for text center line
            5) predictions for disk radii
            6) predictions for cosine(theta)
            7) predictions for sine(theta)
        Sine and cosine are regularized, such that the squared sum equals one.
        """
        f1, f2, f3, f4, f5 = self._backbone(batch)
        h1 = f5
        h2 = self._up1(h1, f4)
        h3 = self._up2(h2, f3)
        h4 = self._up3(h3, f2)
        h5 = self._up4(h4, f1)
        h5_deconv = self._final_deconv(h5)
        pseudo_predictions = self._predict(h5_deconv)

        # Construct new output tensor. If we would not do this and modify the
        # pseudo_predictions inplace in the regularization below, we would cause
        # a RunTimeError from PyTorch during gradient computation.
        output = torch.zeros_like(pseudo_predictions)
        output[:, :5] = pseudo_predictions[:, :5]

        # Regularizing cosθ and sinθ so that the squared sum equals 1. Because
        # then in matches the format of ground truth and we do not need to do
        # this in the loss function.
        scale = torch.sqrt(1. / (torch.pow(pseudo_predictions[:, 5], 2) + torch.pow(pseudo_predictions[:, 6], 2)))
        output[:, 5] = pseudo_predictions[:, 5] * scale  # cosine
        output[:, 6] = pseudo_predictions[:, 6] * scale  # sine

        # We do not apply softmax to tr and tcl pseudo predictions here, because
        # this is done by torch.nn.functional.cross_entropy in the loss
        # function.

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
