from functools import partial
from typing import Tuple, Union, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F


def create_conv(in_channels: int,
                out_channels: int,
                kernel_size: Union[int, Tuple[int, int, int]],
                order: str,
                num_groups: int,
                padding: Union[str, int, Tuple[int, int, int]],
                recurrent: bool = False) -> List[Tuple[str, nn.Module]]:
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        recurrent (bool): use recurrent convolutions

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in "ryek", "Non-linearity cannot be the first operation in the layer"
    if not (recurrent and 'r' not in order):
        raise NotImplementedError("Recurrent convolutions must be used with ReLU non-linearity")

    # Remove ReLU since it's applied in the recurrent convolution
    if recurrent:
        order = order.replace('r', '')

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'y':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'k':
            modules.append(('GELU', nn.GELU()))
        elif char == 'c':
            bias = not any(c in order for c in "gbil") # add learnable bias only in the absence of normalization
            if recurrent:
                modules.append(('rconv', RecurrentConv3d(in_channels,
                                                         out_channels,
                                                         kernel_size,
                                                         padding=padding,
                                                         bias=bias)))
            else:
                modules.append(('conv', nn.Conv3d(in_channels,
                                                  out_channels,
                                                  kernel_size,
                                                  padding=padding,
                                                  bias=bias)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char in 'bil':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                if char == 'b':
                    modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
                elif char == 'i':
                    modules.append(('instancenorm', nn.InstanceNorm3d(in_channels, affine=True)))
                if char == 'l':
                    modules.append(('layernorm', nn.LayerNorm(in_channels)))
            else:
                if char == 'b':
                    modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
                elif char == 'i':
                    modules.append(('instancenorm', nn.InstanceNorm3d(out_channels, affine=True)))
                if char == 'l':
                    modules.append(('layernorm', nn.LayerNorm(out_channels)))
        elif char == 'd':
            modules.append(('dropout', nn.Dropout3d(p=0.1, inplace=True)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['g', 'b', 'i', 'l', 'r', 'y', 'e', 'k', 'c', 'd']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]] = 3,
                 order: str = "gcr",
                 num_groups: int = 8,
                 padding: Union[str, int, Tuple[int, int, int]] = 1):
        super().__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 encoder: bool,
                 kernel_size: Union[int, Tuple[int, int, int]] = 3,
                 order: str = "gcr",
                 num_groups: int = 8,
                 padding: Union[str, int, Tuple[int, int, int]] = 1):
        super().__init__()

        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels,
                                   conv1_out_channels,
                                   kernel_size, order,
                                   num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels,
                                   conv2_out_channels,
                                   kernel_size,
                                   order,
                                   num_groups,
                                   padding=padding))


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 order: str = "cge",
                 num_groups: int = 8,
                 padding: Union[str, int, Tuple[int, int, int]] = 1,
                 **kwargs):
        super().__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels,
                                out_channels,
                                kernel_size,
                                order,
                                num_groups,
                                padding)
        # residual block
        self.conv2 = SingleConv(out_channels,
                                out_channels,
                                kernel_size,
                                order,
                                num_groups,
                                padding)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'ryek':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels,
                                out_channels,
                                kernel_size,
                                n_order,
                                num_groups,
                                padding)

        # create non-linearity separately
        if 'r' in order:
            self.non_linearity = nn.ReLU(inplace=True)
        elif 'y' in order:
            self.non_linearity = nn.LeakyReLU(inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        elif 'k' in order:
            self.non_linearity = nn.GELU()
        else:
            raise ValueError(f"No known non-linearity in the order string ({order}). Use one of 'r', 'y', 'e' or 'k'.")

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class AttentionBlock(nn.Module):
    """Following https://arxiv.org/abs/1804.03999"""

    def __init__(self, out_channels: int):
        super().__init__()

        self.W_gate = nn.Conv3d(2 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_feat = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.psi = nn.Sequential(
                nn.Conv3d(out_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid()
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: Tensor, feature: Tensor) -> Tensor:
        gate = self.W_gate(gate)
        feature = self.W_feat(feature)
        psi = self.psi(self.relu(gate + feature))
        return feature * psi


class RecurrentConv3d(nn.Module):
    """Following https://arxiv.org/abs/1802.06955."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 steps: int = 2,
                 residual: bool = False):
        super().__init__()

        self.conv3d = nn.Conv3d(out_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                                dilation,
                                groups,
                                bias,
                                padding_mode,
                                device,
                                dtype)

        if in_channels != out_channels:
            self.conv3d_in = nn.Conv3d(in_channels,
                                       out_channels,
                                       kernel_size=1,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=bias,
                                       padding_mode=padding_mode,
                                       device=device,
                                       dtype=dtype)
        else:
            self.conv3d_in = None

        self.steps = steps
        self.residual = residual

    def forward(self, x):
        if self.conv3d_in is not None:
            x = self.conv3d_in(x)
        xi = F.relu(self.conv3d(x), inplace=True)
        for step in range(self.steps - 1):
            xi = F.relu(self.conv3d(x + xi), inplace=True)
        return xi + x if self.residual else xi


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_kernel_size: Union[int, Tuple[int, int, int]] = 3,
                 apply_pooling: bool = True,
                 pool_kernel_size: Union[int, Tuple[int, int, int]] = 2,
                 pool_type: str = "max",
                 basic_module: type(nn.Module) = DoubleConv,
                 conv_layer_order: str = "gcr",
                 num_groups: int = 8,
                 padding: Union[str, int, Tuple[int, int, int]] = 1):
        super().__init__()

        assert pool_type in ["max", "avg"]
        if apply_pooling:
            if pool_type == "max":
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels,
                                         out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation) followed by a basic module (DoubleConv or ExtResNetBlock).
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        upsample_mode (str): upsampling mode, one of 'nearest', 'linear', 'bilinear', 'trilinear', 'area'
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (boole): should the input be upsampled
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_kernel_size: Union[int, Tuple[int, int, int]] = 3,
                 scale_factor: Union[int, Tuple[int, int, int]] = (2, 2, 2),
                 basic_module: type(nn.Module) = DoubleConv,
                 conv_layer_order: str = "gcr",
                 num_groups: int = 8,
                 upsample_mode: str = "nearest",
                 padding: Union[str, int, Tuple[int, int, int]] = 1,
                 upsample: bool = True,
                 attention: bool = False):
        super().__init__()

        if upsample:
            if basic_module == DoubleConv:
                # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
                self.upsampling = InterpolateUpsampling(mode=upsample_mode)
                # concat joining
                self.joining = partial(self._joining, concat=True)
            else:
                # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels,
                                                          out_channels=out_channels,
                                                          kernel_size=conv_kernel_size,
                                                          scale_factor=scale_factor,
                                                          padding=padding)
                # sum joining
                self.joining = partial(self._joining, concat=False)
                # adapt the number of in_channels for the ExtResNetBlock
                in_channels = out_channels
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        if attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = None

        self.basic_module = basic_module(in_channels,
                                         out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        if self.attention is not None:
            encoder_features = self.attention(gate=x, feature=encoder_features)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


def create_encoders(in_channels: int,
                    f_maps: Union[List[int], Tuple[int]],
                    basic_module: type(nn.Module),
                    conv_kernel_size: Union[int, Tuple[int, int, int]],
                    conv_padding: Union[str, int, Tuple[int, int, int]],
                    layer_order: str,
                    num_groups: int,
                    pool_kernel_size: Union[int, Tuple[int, int, int]]) -> nn.ModuleList:
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(in_channels=in_channels,
                              out_channels=out_feature_num,
                              apply_pooling=False,  # skip pooling in the firs encoder
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding)
        else:
            # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
            encoder = Encoder(in_channels=f_maps[i - 1],
                              out_channels=out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              pool_kernel_size=pool_kernel_size,
                              padding=conv_padding)

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def create_decoders(f_maps: Union[List[int], Tuple[int]],
                    basic_module: type(nn.Module),
                    conv_kernel_size: Union[int, Tuple[int, int, int]],
                    conv_padding: Union[str, int, Tuple[int, int, int]],
                    layer_order: str,
                    num_groups: int,
                    upsample_mode: str,
                    upsample: bool,
                    attention: bool) -> nn.ModuleList:
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
        # currently strides with a constant stride: (2, 2, 2)

        _upsample = True
        if i == 0:
            # upsampling can be skipped only for the 1st decoder, afterwards it should always be present
            _upsample = upsample

        decoder = Decoder(in_channels=in_feature_num,
                          out_channels=out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          upsample_mode=upsample_mode,
                          padding=conv_padding,
                          upsample=_upsample,
                          attention=attention)
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode: str = "nearest"):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True

    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]] = 3,
                 scale_factor: Union[int, Tuple[int, int, int]] = (2, 2, 2),
                 padding: Union[str, int, Tuple[int, int, int]] = 1):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        upsample = nn.ConvTranspose3d(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      stride=scale_factor,
                                      padding=padding)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x
