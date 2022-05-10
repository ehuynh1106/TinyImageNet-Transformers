from fileio import get_label_mapping
from torchvision.transforms import Normalize

import random, torch
import matplotlib.pyplot as plt

def cut_long_label(str):
    str = str.split(",")
    if len(str) >= 3:
        str = f"{str[0]},{str[1]},{str[2]},..."
    else:
        str = ",".join(str)
    return str

@torch.no_grad()
def visualize(x, pred, y, epoch):
    batch_size = len(y)
    idx = random.randint(0, batch_size-1)

    prediction = torch.max(pred.data, 1)[1][idx].item()
    actual = y[idx].item()
    inv_normalize = Normalize(
        mean=(-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
        std=(1/0.229, 1/0.224, 1/0.225)
    )
    random_sample = torch.round(torch.mul(inv_normalize(x[idx]), 225)).type(torch.ByteTensor).cpu().permute(1,2,0)

    mapping = get_label_mapping()
    plt.imshow(random_sample)
    label_str = cut_long_label(mapping.iloc[actual]['label'])
    prediction_str = cut_long_label(mapping.iloc[prediction]['label'])

    plt.title(f"Label: {label_str}\nPrediction: {prediction_str}", wrap=True, fontsize=12)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f'predictions/val{epoch+1}_{idx}.png')