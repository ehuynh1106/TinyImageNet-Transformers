"""
Taken from timm (https://github.com/rwightman/pytorch-image-models) and
modified to work with gradient accumulation
"""

import torch
from timm.utils.clip_grad import dispatch_clip_grad

class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer,
        clip_grad=None, clip_mode='norm', parameters=None,
        create_graph=False, update_grad=False
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        if update_grad:
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)