from typing import Union, Tuple, Dict, Any

import torch.nn as nn

from pytorch3dunet.unet3d.buildingblocks import DoubleConv, ExtResNetBlock, create_encoders, create_decoders
from pytorch3dunet.unet3d.utils import number_of_features_per_level, get_class


class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        upsample_mode (str): upsampling mode, one of 'nearest', 'linear', 'bilinear', 'trilinear', 'area'
        attention (bool): if True add attention module to the decoder path
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 final_sigmoid: bool,
                 basic_module: type(nn.Module),
                 f_maps: int = 64,
                 layer_order: str = "gcr",
                 num_groups: int = 8,
                 num_levels: int = 4,
                 is_segmentation: bool = True,
                 conv_kernel_size: Union[int, Tuple[int, int, int]] = 3,
                 pool_kernel_size: Union[int, Tuple[int, int, int]] = 2,
                 conv_padding: Union[str, int, Tuple[int, int, int]] = 1,
                 upsample_mode: str = "nearest",
                 attention: bool = False,
                 **kwargs: Any):
        super().__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels,
                                        f_maps,
                                        basic_module,
                                        conv_kernel_size,
                                        conv_padding,
                                        layer_order,
                                        num_groups,
                                        pool_kernel_size)

        # create decoder path
        self.decoders = create_decoders(f_maps,
                                        basic_module,
                                        conv_kernel_size,
                                        conv_padding,
                                        layer_order,
                                        num_groups,
                                        upsample_mode,
                                        upsample=True,
                                        attention=attention)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` or `ExtResNetBlock` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 final_sigmoid: bool = True,
                 f_maps: int = 64,
                 layer_order: str = "gcr",
                 num_groups: int = 8,
                 num_levels: int = 4,
                 is_segmentation: bool = True,
                 conv_padding: Union[str, int, Tuple[int, int, int]] = 1,
                 residual: bool = False,
                 attention: bool = False,
                 **kwargs: Any):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         final_sigmoid=final_sigmoid,
                         basic_module=ExtResNetBlock if residual else DoubleConv,
                         f_maps=f_maps,
                         layer_order=layer_order,
                         num_groups=num_groups,
                         num_levels=num_levels,
                         is_segmentation=is_segmentation,
                         conv_padding=conv_padding,
                         attention=attention,
                         **kwargs)


class UNet2D(Abstract3DUNet):
    """
    Just a standard 2D Unet. Arises naturally by specifying conv_kernel_size=(1, 3, 3), pool_kernel_size=(1, 2, 2).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 final_sigmoid: bool = True,
                 f_maps: int = 64,
                 layer_order: str = "gcr",
                 num_groups: int = 8,
                 num_levels: int = 4,
                 is_segmentation: bool = True,
                 conv_padding: Union[str, int, Tuple[int, int, int]] = 1,
                 residual: bool = False,
                 attention: bool = False,
                 **kwargs: Any):
        if isinstance(conv_padding, int):
            conv_padding = (0, conv_padding, conv_padding)
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         final_sigmoid=final_sigmoid,
                         basic_module=ExtResNetBlock if residual else DoubleConv,
                         f_maps=f_maps,
                         layer_order=layer_order,
                         num_groups=num_groups,
                         num_levels=num_levels,
                         is_segmentation=is_segmentation,
                         conv_kernel_size=(1, 3, 3),
                         pool_kernel_size=(1, 2, 2),
                         conv_padding=conv_padding,
                         attention=attention,
                         **kwargs)


def get_model(model_config: Dict) -> nn.Module:
    model_class = get_class(model_config["name"], modules=["pytorch3dunet.unet3d.model"])
    return model_class(**model_config)
