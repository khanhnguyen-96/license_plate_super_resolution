import torch

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6 # (1e-3)^2

    def forward(self, input, target):
        _assert_no_grad(target)

        error = torch.sqrt( (input-target)**2 + self.eps )
        return error.mean()
