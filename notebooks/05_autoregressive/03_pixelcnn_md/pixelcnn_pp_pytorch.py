"""A compact PixelCNN++ in PyTorch - a stand-in for tfp.distributions.PixelCNN.

PyTorch has no ready-made PixelCNN distribution, so this file implements the
main ideas of PixelCNN++ (Salimans et al., 2017), which is also what
TensorFlow Probability's PixelCNN is based on:

  * Two stacks of "shifted" convolutions instead of masked ones: the vertical
    stack `u` only sees rows above the current pixel, the horizontal stack `ul`
    sees rows above plus pixels to the left. This removes the blind spot of
    the original masked PixelCNN.
  * Gated residual blocks with concat_elu activations and dropout.
  * A U-Net-like structure with `num_hierarchies` resolutions (downsampling
    with strided convs, upsampling with transposed convs, skip connections).
  * Each pixel is modelled by a mixture of `num_logistic_mix` logistic
    distributions, discretised into 256 bins (the "discretized logistic mixture
    likelihood"), instead of a 256-way softmax.

Simplifications compared with TFP: only single-channel (grayscale) images,
no weight normalisation and no conditional inputs.

`PixelCNN` mirrors the TFP API used in the notebook: `log_prob(images)` and
`sample(n)`, with pixel values in [low, high] = [0, 255].
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def concat_elu(x):
    """ELU of both x and -x, concatenated on the channel axis (doubles channels)."""
    return F.elu(torch.cat([x, -x], dim=1))


def down_shift(x):
    """Shift the image one row down (zero row on top), so a pixel cannot see its own row."""
    return F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]


def right_shift(x):
    """Shift the image one column right (zero column on the left)."""
    return F.pad(x, (1, 0, 0, 0))[:, :, :, :-1]


class DownShiftedConv2d(nn.Conv2d):
    """Conv whose output at (i, j) only sees rows <= i (padding on top, none below)."""

    def __init__(self, in_channels, out_channels, kernel_size=(2, 3), stride=1):
        super().__init__(in_channels, out_channels, kernel_size, stride)
        kh, kw = kernel_size
        self.shift_pad = ((kw - 1) // 2, (kw - 1) // 2, kh - 1, 0)  # left, right, top, bottom

    def forward(self, x):
        return super().forward(F.pad(x, self.shift_pad))


class DownRightShiftedConv2d(nn.Conv2d):
    """Conv whose output at (i, j) only sees rows <= i and columns <= j."""

    def __init__(self, in_channels, out_channels, kernel_size=(2, 2), stride=1):
        super().__init__(in_channels, out_channels, kernel_size, stride)
        kh, kw = kernel_size
        self.shift_pad = (kw - 1, 0, kh - 1, 0)

    def forward(self, x):
        return super().forward(F.pad(x, self.shift_pad))


class DownShiftedDeconv2d(nn.ConvTranspose2d):
    """2x upsampling that keeps the 'rows above' property (crops the extra border)."""

    def __init__(self, in_channels, out_channels, kernel_size=(2, 3), stride=2):
        super().__init__(in_channels, out_channels, kernel_size, stride, output_padding=1)

    def forward(self, x):
        x = super().forward(x)
        kh, kw = self.kernel_size
        return x[:, :, : x.shape[2] - kh + 1, (kw - 1) // 2 : x.shape[3] - (kw - 1) // 2]


class DownRightShiftedDeconv2d(nn.ConvTranspose2d):
    """2x upsampling that keeps the 'above and to the left' property."""

    def __init__(self, in_channels, out_channels, kernel_size=(2, 2), stride=2):
        super().__init__(in_channels, out_channels, kernel_size, stride, output_padding=1)

    def forward(self, x):
        x = super().forward(x)
        kh, kw = self.kernel_size
        return x[:, :, : x.shape[2] - kh + 1, : x.shape[3] - kw + 1]


class GatedResNet(nn.Module):
    """Residual block: x + a * sigmoid(b), optionally conditioned on `n_aux` skip inputs."""

    def __init__(self, num_filters, conv_class, n_aux=0, dropout_p=0.0):
        super().__init__()
        self.conv_in = conv_class(2 * num_filters, num_filters)
        # 1x1 conv ("network in network") that mixes in the skip connection(s)
        self.aux = nn.Conv2d(2 * n_aux * num_filters, num_filters, 1) if n_aux else None
        self.dropout = nn.Dropout(dropout_p)
        self.conv_out = conv_class(2 * num_filters, 2 * num_filters)

    def forward(self, x, aux=None):
        h = self.conv_in(concat_elu(x))
        if aux is not None:
            h = h + self.aux(concat_elu(aux))
        h = self.dropout(concat_elu(h))
        a, b = self.conv_out(h).chunk(2, dim=1)
        return x + a * torch.sigmoid(b)  # gated residual connection


# ---------------------------------------------------------------------------
# The network: image in [-1, 1] -> mixture parameters for every pixel
# ---------------------------------------------------------------------------
class PixelCNNNetwork(nn.Module):
    def __init__(self, num_resnet, num_hierarchies, num_filters, num_logistic_mix, dropout_p):
        super().__init__()
        nf, p = num_filters, dropout_p

        def u_block(n_aux):
            return GatedResNet(nf, DownShiftedConv2d, n_aux, p)

        def ul_block(n_aux):
            return GatedResNet(nf, DownRightShiftedConv2d, n_aux, p)

        # First layers: shifted so that no pixel sees itself
        self.u_init = DownShiftedConv2d(1, nf, (2, 3))
        self.ul_init_up = DownShiftedConv2d(1, nf, (1, 3))
        self.ul_init_left = DownRightShiftedConv2d(1, nf, (2, 1))

        # Up pass: `num_resnet` blocks per resolution, then halve the resolution
        self.up_u, self.up_ul = nn.ModuleList(), nn.ModuleList()
        self.downsample_u, self.downsample_ul = nn.ModuleList(), nn.ModuleList()
        for level in range(num_hierarchies):
            self.up_u.append(nn.ModuleList([u_block(0) for _ in range(num_resnet)]))
            # the horizontal stack is conditioned on the vertical stack
            self.up_ul.append(nn.ModuleList([ul_block(1) for _ in range(num_resnet)]))
            if level < num_hierarchies - 1:
                self.downsample_u.append(DownShiftedConv2d(nf, nf, (2, 3), stride=2))
                self.downsample_ul.append(DownRightShiftedConv2d(nf, nf, (2, 2), stride=2))

        # Down pass (coarse -> fine): every block consumes one skip connection
        self.down_u, self.down_ul = nn.ModuleList(), nn.ModuleList()
        self.upsample_u, self.upsample_ul = nn.ModuleList(), nn.ModuleList()
        for level in range(num_hierarchies):
            n_blocks = num_resnet if level == 0 else num_resnet + 1
            self.down_u.append(nn.ModuleList([u_block(1) for _ in range(n_blocks)]))
            self.down_ul.append(nn.ModuleList([ul_block(2) for _ in range(n_blocks)]))
            if level < num_hierarchies - 1:
                self.upsample_u.append(DownShiftedDeconv2d(nf, nf))
                self.upsample_ul.append(DownRightShiftedDeconv2d(nf, nf))

        # Per pixel and mixture component: a mixture logit, a location and a log-scale
        self.out = nn.Conv2d(nf, 3 * num_logistic_mix, 1)

    def forward(self, x):
        u = down_shift(self.u_init(x))
        ul = down_shift(self.ul_init_up(x)) + right_shift(self.ul_init_left(x))
        us, uls = [u], [ul]  # skip connections

        for level in range(len(self.up_u)):
            for u_blk, ul_blk in zip(self.up_u[level], self.up_ul[level]):
                u = u_blk(u)
                ul = ul_blk(ul, aux=u)
                us.append(u)
                uls.append(ul)
            if level < len(self.downsample_u):
                u = self.downsample_u[level](u)
                ul = self.downsample_ul[level](ul)
                us.append(u)
                uls.append(ul)

        u, ul = us.pop(), uls.pop()
        for level in range(len(self.down_u)):
            for u_blk, ul_blk in zip(self.down_u[level], self.down_ul[level]):
                u = u_blk(u, aux=us.pop())
                ul = ul_blk(ul, aux=torch.cat([u, uls.pop()], dim=1))
            if level < len(self.upsample_u):
                u = self.upsample_u[level](u)
                ul = self.upsample_ul[level](ul)

        return self.out(F.elu(ul))


# ---------------------------------------------------------------------------
# The discretized logistic mixture likelihood
# ---------------------------------------------------------------------------
def discretized_logistic_mix_log_prob(x, params, num_levels=256):
    """Log-probability of every pixel of x (in [-1, 1]) under the predicted mixtures.

    Pixel values are discrete, so each value owns a bin of width 2 / (num_levels - 1)
    and its probability is CDF(bin top) - CDF(bin bottom) of each logistic
    component. The lowest / highest bins extend to -inf / +inf.
    Returns a tensor of shape (batch, H, W).
    """
    logit_probs, means, log_scales = params.chunk(3, dim=1)  # each (B, K, H, W)
    log_scales = log_scales.clamp(min=-7.0)
    half_bin = 1.0 / (num_levels - 1)

    centered = x - means  # x (B, 1, H, W) broadcasts over the K components
    inv_scale = torch.exp(-log_scales)
    plus_in = inv_scale * (centered + half_bin)
    min_in = inv_scale * (centered - half_bin)

    log_cdf_plus = F.logsigmoid(plus_in)  # log CDF(top), used for the lowest bin
    log_one_minus_cdf_min = -F.softplus(min_in)  # log(1 - CDF(bottom)), highest bin
    cdf_delta = torch.sigmoid(plus_in) - torch.sigmoid(min_in)  # mass of the bin
    # For numerical safety, a tiny bin mass is approximated by density * bin width
    mid_in = inv_scale * centered
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
    log_bin = torch.where(
        cdf_delta > 1e-5,
        torch.log(cdf_delta.clamp(min=1e-12)),
        log_pdf_mid - math.log((num_levels - 1) / 2),
    )

    log_probs = torch.where(
        x < -0.999, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, log_bin)
    )
    # Mixture: log sum_k pi_k * p_k(x), computed stably with logsumexp
    return torch.logsumexp(log_probs + F.log_softmax(logit_probs, dim=1), dim=1)


# ---------------------------------------------------------------------------
# The distribution object used in the notebook (like tfp.distributions.PixelCNN)
# ---------------------------------------------------------------------------
class PixelCNN(nn.Module):
    def __init__(
        self,
        image_shape,  # (channels, height, width) - channels first, PyTorch style
        num_resnet=5,
        num_hierarchies=3,
        num_filters=160,
        num_logistic_mix=10,
        dropout_p=0.5,
        high=255,
        low=0,
    ):
        super().__init__()
        assert image_shape[0] == 1, "this compact version supports grayscale images only"
        self.image_shape = tuple(image_shape)
        self.high, self.low = high, low
        self.network = PixelCNNNetwork(
            num_resnet, num_hierarchies, num_filters, num_logistic_mix, dropout_p
        )

    def _to_unit(self, images):
        """[low, high] -> [-1, 1]"""
        return 2.0 * (images - self.low) / (self.high - self.low) - 1.0

    def log_prob(self, images):
        """Log-likelihood of each image (summed over pixels), shape (batch,)."""
        x = self._to_unit(images)
        params = self.network(x)
        num_levels = self.high - self.low + 1
        return discretized_logistic_mix_log_prob(x, params, num_levels).sum(dim=(1, 2))

    @torch.no_grad()
    def sample(self, n):
        """Draw n images, one pixel at a time (each pixel depends on the previous ones)."""
        was_training = self.training
        self.eval()  # no dropout while sampling
        device = next(self.parameters()).device
        num_levels = self.high - self.low + 1
        x = torch.zeros((n,) + self.image_shape, device=device)  # in [-1, 1]
        _, height, width = self.image_shape

        for row in range(height):
            for col in range(width):
                params = self.network(x)[:, :, row, col]  # (n, 3K)
                logit_probs, means, log_scales = params.chunk(3, dim=1)
                # 1. choose a mixture component for every image
                k = torch.distributions.Categorical(logits=logit_probs).sample()
                mean = means.gather(1, k[:, None]).squeeze(1)
                log_scale = log_scales.gather(1, k[:, None]).squeeze(1).clamp(min=-7.0)
                # 2. sample that logistic with the inverse CDF: mean + s * log(u / (1 - u))
                u = torch.rand_like(mean).clamp(1e-5, 1.0 - 1e-5)
                value = mean + torch.exp(log_scale) * (torch.log(u) - torch.log1p(-u))
                # 3. snap to the nearest of the discrete pixel values
                value = torch.round((value.clamp(-1, 1) + 1) / 2 * (num_levels - 1))
                x[:, 0, row, col] = value / (num_levels - 1) * 2 - 1

        self.train(was_training)
        return (x + 1) / 2 * (self.high - self.low) + self.low  # back to [low, high]
