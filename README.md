# Vision Transformers in 2022: An Update on Tiny ImageNet
This is the official PyTorch repository of [Vision Transformers in 2022: An Update on Tiny ImageNet](report.pdf) with pretrained models and training and evaluation scripts.

# Model Zoo
I provide the following models finetuned with a 384x384 image resolution on Tiny ImageNet.

| name | acc@1 | #params | url |
| --- | --- | --- | --- |
| ViT-L | 86.31 | 304M | [model](https://drive.google.com/file/d/1_3h4WhBsVc6PWgCqwWT5uOnzPxkD17Pn/view?usp=sharing) |
| CaiT-S36 | 86.64 | 68M | [model](https://drive.google.com/file/d/1JrMWc_iuhz4JjOy_qc265UVEyhUG1cOK/view?usp=sharing) |
| DeiT-B distilled | 87.35 | 87M | [model](https://drive.google.com/file/d/1JMFQMN1GfbUjcsqdDipazJn8CyJJ3u_p/view?usp=sharing) |
| Swin-L | 91.35 | 195M | [model](https://drive.google.com/file/d/15YpRF0lFWAfoIto4FaQ0a-Y5pqusp3Zi/view?usp=sharing) |

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