# Vision Transformers in 2022: An Update on Tiny ImageNet
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformers-in-2022-an-update-on-tiny/image-classification-on-tiny-imagenet-1)](https://paperswithcode.com/sota/image-classification-on-tiny-imagenet-1?p=vision-transformers-in-2022-an-update-on-tiny)

This is the official PyTorch repository of [Vision Transformers in 2022: An Update on Tiny ImageNet](https://arxiv.org/abs/2205.10660) with pretrained models and training and evaluation scripts.

# Model Zoo
I provide the following models finetuned with a 384x384 image resolution on Tiny ImageNet.

| name | acc@1 | #params | url |
| --- | --- | --- | --- |
| ViT-L | 86.43 | 304M | [model](https://github.com/ehuynh1106/TinyImageNet-Transformers/releases/download/weights/vit_large_384.pth) |
| CaiT-S36 | 86.74 | 68M | [model](https://github.com/ehuynh1106/TinyImageNet-Transformers/releases/download/weights/cait_s36_384.pth) |
| DeiT-B distilled | 87.29 | 87M | [model](https://github.com/ehuynh1106/TinyImageNet-Transformers/releases/download/weights/deit_base_distilled_384.pth) |
| Swin-L | 91.35 | 195M | [model](https://github.com/ehuynh1106/TinyImageNet-Transformers/releases/download/weights/swin_large_384.pth) |

# Usage

First, clone the repository:
```
git clone https://github.com/ehuynh1106/TinyImageNet-Transformers.git
```

Then install the dependencies:
```
pip install -r requirements.txt
```

# Data Preparation
Download and extract Tiny ImageNet at [https://image-net.org/](https://image-net.org/) in the home directory of this repository.

Then run `python fileio.py` to format the data. This will convert the images into tensors and pickle them into two files, `train_dataset.pkl` and `val_dataset.pkl` that will be used in the main code.

# Training
To train a Swin-L model on Tiny ImageNet run the following command:
```
python main.py --train --model swin
```
Note: Training checkpoints are automatically saved in `/models` and visualizations of predictions on the validation set are automically saved to `/predictions` after half of the epochs have passed.

To train DeiT, ViT, and CaiT, replace `--model swin` with `--model deit/vit/cait`.

To resume training a Swin-L model on Tiny ImageNet run the following command:
```
python main.py --train --model swin --resume /path/to/checkpoint
```

# Evaluate
To evaluate a Swin-L model on the validation set of Tiny ImageNet run the following command:
```
python main.py --evaluate /path/to/model --model swin
```

# Citing
```bibtex
@misc{huynh2022vision,
      title={Vision Transformers in 2022: An Update on Tiny ImageNet}, 
      author={Ethan Huynh},
      year={2022},
      eprint={2205.10660},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```