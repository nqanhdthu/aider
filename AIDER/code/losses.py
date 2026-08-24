import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma=gamma
    def forward(self, logits, targets):
        ce=F.cross_entropy(logits,targets,reduction="none")
        pt=torch.exp(-ce)
        return ((1.0-pt)**self.gamma * ce).mean()

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts):
        super().__init__()
        counts=torch.as_tensor(class_counts,dtype=torch.float32)
        if torch.any(counts<=0):
            raise ValueError("All training class counts must be positive.")
        self.register_buffer("log_counts",torch.log(counts))
    def forward(self, logits, targets):
        return F.cross_entropy(logits+self.log_counts,targets)

class SupConLoss(nn.Module):
    """Standard supervised contrastive loss for 2+ views per sample."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature=temperature
    def forward(self, features, labels):
        # features: [batch, n_views, dim], assumed L2 normalized
        b,v,d=features.shape
        z=features.reshape(b*v,d)
        labels=labels.reshape(-1,1)
        mask=(labels==labels.T).float().to(z.device)
        mask=mask.repeat_interleave(v,0).repeat_interleave(v,1)
        logits=(z @ z.T)/self.temperature
        eye=torch.eye(b*v,device=z.device)
        logits_mask=1.0-eye
        logits=logits-logits.max(dim=1,keepdim=True).values.detach()
        exp_logits=torch.exp(logits)*logits_mask
        log_prob=logits-torch.log(exp_logits.sum(dim=1,keepdim=True)+1e-12)
        pos_mask=mask*logits_mask
        denom=pos_mask.sum(dim=1).clamp_min(1.0)
        return -((pos_mask*log_prob).sum(dim=1)/denom).mean()
