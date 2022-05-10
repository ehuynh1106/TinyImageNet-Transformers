from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing

def get_data_augmentations(label_smoothing, en_mixup=True, en_cutmix=True, en_randerase=True):
    mixup=None
    random_erase=None
    m_alpha = 0.8 if en_mixup else 0
    c_alpha = 1.0 if en_cutmix else 0
    if en_mixup or en_cutmix:
        mixup = Mixup(
            mixup_alpha=m_alpha,
            cutmix_alpha=c_alpha,
            label_smoothing=0.1,
            num_classes=200
        )
    if en_randerase:
        random_erase=RandomErasing(
            probability=0.25,
            mode='pixel'
        )
    
    return mixup, random_erase