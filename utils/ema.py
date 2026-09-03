import copy
import torch


class ModelEMA:
    """
    Exponential Moving Average (EMA) parameter tracker.
    
    Maintains a smoothed shadow copy of model weights during training,
    improving generalization and stabilizing validation curves.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model: torch.nn.Module):
        """
        Updates shadow parameters with current model weights.
        """
        with torch.no_grad():
            msd = model.state_dict()
            for k in self.shadow.keys():
                if self.shadow[k].dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
                else:
                    self.shadow[k].copy_(msd[k])
