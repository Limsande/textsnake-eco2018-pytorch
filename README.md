# Implementing TextSnakes for minirhizotron images in PyTorch

An implementation of plant root detection in minirhizotron images inspired by
[TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes](https://arxiv.org/abs/1807.01544)
(Long et al. 2020).

TextSnake is a method to represent and extract curved text from images. This repository is an adaptation of this concept
to plant roots.

NOTE that this is a proof of concept, written to work with the unpublished dataset Eco2018. However, it should be as
easy as writing a custom DataLoader and providing appropriate input to use different datasets. The
[TextSnake.pytorch](https://github.com/princewang1994/TextSnake.pytorch) repo, for example, already contains code to
precompute all required input from annotated images.

ALSO NOTE that the geometry loss is currently bugged and the model won't learn. The option `--no-geometry-loss` is a
temporary workaround for that.


## Branches
* `master`: A version of the code that is compatible with torch 1.8
* `torch1.4`: A version of the code that is compatible with torch 1.4

## Usage
The script `reproduce_paper.py` recreates the experiment of Long _et al._ on the Total-Text dataset.


```
usage: main.py [-h] --lr LR --epochs N [--batch-size N] [--num-workers N]
               [--val-interval N] [--resume FILE] [--pretrained-backbone]
               [--no-geometry-loss]

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate
  --epochs N            Number of epochs
  --batch-size N        Batch size (default 1)
  --num-workers N       Number of processes used to load data
  --val-interval N      Evaluate model after each N epochs (default 1)
  --resume FILE         Resume training at checkpoint loaded from FILE
  --pretrained-backbone
                        Use as backbone a VGG16 pretrained on ImageNet from
                        the torchvision GitHub repo
  --no-geometry-loss    Ignore geometry loss during training
```


## Requirements
- torch
- torchvision
- opencv
- scikit-image
- scipy

See [requirements.txt](requirements.txt) for an extensive list of packages and versions. There is also a
[conda environment](conda-env.yaml), including the latest version of Python this code was tested with. 


## Todo

- [ ] Fix geometry loss
- [ ] Implement pretraining on SynthText
- [ ] Implement evaluation on the test set


## License
This repository is licensed under the [MIT License](LICENSE.md).

The code in `dataloader/TotalTextLoader.py` is from the [TextSnake.pytorch](https://github.com/princewang1994/TextSnake.pytorch)
repository of GitHub user _princewang1994_, which is also licensed under the MIT License.
