# Implemented by Nguyen Tran Toan (trantoan060689@gmail.com). Oct 04 2018.
import torch.nn as nn

def pixel_split(input, scale_factor):
    r"""Rearranges elements in a tensor of shape :math:`[*, C, H, W]` to a
    tensor of shape :math:`[C*r^2, H/r, W/r]`.

    See :class:`~torch.nn.PixelSplit` for details.

    Args:
        input (Tensor): Input
        scale_factor (int): factor to decrease spatial resolution by

    Examples::

        >>> ps = nn.PixelSplit(3)
        >>> input = torch.empty(1, 3, 12, 12)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 27, 4, 4])
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // scale_factor
    out_width = in_width // scale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, scale_factor,
        out_width, scale_factor)

    splitted = input_view.permute(0, 1, 5, 3, 2, 4).contiguous()

    channels *= scale_factor ** 2

    return splitted.view(batch_size, channels, out_height, out_width)

class PixelSplit(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(*, C, H, W)` to a
    tensor of shape :math:`(Cr^2, H/r, W/r)`.

    Args:
        scale_factor (int): factor to decrease spatial resolution by

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C * \text{scale_factor}^2, H / \text{scale_factor}, W / \text{scale_factor})`

    Examples::

        >>> ps = nn.PixelSplit(3)
        >>> input = torch.tensor(1, 3, 12, 12)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 27, 4, 4])
    """

    def __init__(self, scale_factor):
        super(PixelSplit, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return pixel_split(input, self.scale_factor)

    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)
