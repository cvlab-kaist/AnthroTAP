import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange



class Conv2dSamePadding(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
      return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      ih, iw = x.size()[-2:]

      pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
      pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

      if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        )
      return F.conv2d(
        x,
        self.weight,
        self.bias,
        self.stride,
        # self.padding,
        0,
        self.dilation,
        self.groups,
      )


def prepare_output(outputs, num_frames, reduce_mean):
    outputs = rearrange(outputs, '(b t) c h w -> b c t h w', t=num_frames)
    if reduce_mean:
        outputs = outputs.mean(dim=(-1, -2, -3))
    return outputs


def temporal_shift(x, num_frames, channel_shift_fraction):
    x = rearrange(x, '(b t) c h w -> b c t h w', t=num_frames)
    n_channels = x.shape[1]
    n_shift = int(n_channels * channel_shift_fraction)

    shifted_backward = x[:, -n_shift:, 1:, :, :]
    shifted_backward = F.pad(shifted_backward, (0, 0, 0, 0, 0, 1, 0, 0, 0, 0))
    shifted_forward = x[:, :n_shift, :-1, :, :]
    shifted_forward = F.pad(shifted_forward, (0, 0, 0, 0, 1, 0, 0, 0, 0, 0))

    no_shift = x[:, n_shift:-n_shift, :, :, :]
    x = torch.cat([shifted_backward, no_shift, shifted_forward], dim=1)
    x = rearrange(x, 'b c t h w -> (b t) c h w')
    return x
    

class TSMResNetBlock(nn.Module):
    def __init__(
      self,
      input_channels,
      output_channels,
      stride,
      use_projection,
      tsm_mode,
      channel_shift_fraction = 0.125,
      rate = 1,
      use_bottleneck = False,
      normalize_fn = 'batchnorm',
    ):
        super().__init__()

        self._input_channels = input_channels if use_bottleneck else input_channels // 4
        self._output_channels = (
            output_channels if use_bottleneck else output_channels // 4)
        self._bottleneck_channels = output_channels // 4
        self._stride = stride
        self._rate = rate
        self._use_projection = use_projection
        self._tsm_mode = tsm_mode
        self._channel_shift_fraction = channel_shift_fraction
        self._use_bottleneck = use_bottleneck

        if self._use_projection:
            self.projection = Conv2dSamePadding(
                self._input_channels, self._output_channels, kernel_size=1, stride=self._stride, bias=False)

        if normalize_fn == 'batchnorm':
            self.norm_0 = nn.BatchNorm2d(self._input_channels)
            if self._use_bottleneck:
                self.norm_1 = nn.BatchNorm2d(self._bottleneck_channels)
            self.norm_2 = nn.BatchNorm2d(self._bottleneck_channels)
        elif normalize_fn == 'groupnorm':
            self.norm_0 = nn.GroupNorm(self._input_channels // 16, self._input_channels)
            if self._use_bottleneck:
                self.norm_1 = nn.GroupNorm(self._input_channels // 16, self._bottleneck_channels)
            self.norm_2 = nn.GroupNorm(self._input_channels // 16, self._bottleneck_channels)


        self.conv_0 = Conv2dSamePadding(
            self._input_channels,
            self._bottleneck_channels,
            kernel_size=1 if self._use_bottleneck else 3,
            stride=1 if self._use_bottleneck else self._stride,
            padding=0 if self._use_bottleneck else 1,
            bias=False,
        )
        if self._use_bottleneck:
            self.conv_1 = Conv2dSamePadding(
                self._bottleneck_channels,
                self._bottleneck_channels,
                kernel_size=3,
                stride=self._stride,
                padding=1,
                dilation=self._rate,
                bias=False,
            )
        self.conv_2 = Conv2dSamePadding(
            self._bottleneck_channels,
            self._output_channels,
            kernel_size=1 if self._use_bottleneck else 3,
            stride=1,
            padding=0 if self._use_bottleneck else 1,
            bias=False,
        )

    def forward(self, inputs, num_frames):
        preact = inputs
        preact = self.norm_0(preact)
        preact = F.relu(preact)

        if self._use_projection:
            shortcut = self.projection(preact)
        else:
            shortcut = inputs
        
        if self._channel_shift_fraction != 0:
            preact = temporal_shift(
                preact,
                num_frames=num_frames,
                channel_shift_fraction=self._channel_shift_fraction,
            )
        
        residual = self.conv_0(preact)

        if self._use_bottleneck:
            residual = self.norm_1(residual)
            residual = F.relu(residual)
            residual = self.conv_1(residual)
        
        residual = self.norm_2(residual)
        residual = F.relu(residual)
        residual = self.conv_2(residual)

        return shortcut + residual

class TSMResNetUnit(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        num_blocks,
        stride,
        tsm_mode,
        normalize_fn = None,
        channel_shift_fraction = 0.125,
        rate = 1,
        use_bottleneck = False,
    ):
        super().__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._num_blocks = num_blocks
        self._normalize_fn = normalize_fn
        self._stride = stride
        self._tsm_mode = tsm_mode
        self._channel_shift_fraction = channel_shift_fraction
        self._rate = rate
        self._use_bottleneck = use_bottleneck

        self.blocks = nn.ModuleList()
        for i in range(self._num_blocks):
            self.blocks.append(
                TSMResNetBlock(
                    self._input_channels if i == 0 else self._output_channels,
                    self._output_channels,
                    stride=self._stride if i == 0 else 1,
                    rate=(max(self._rate // 2, 1) if i == 0 else self._rate),
                    use_projection=(i == 0),
                    tsm_mode=self._tsm_mode,
                    channel_shift_fraction=self._channel_shift_fraction,
                    normalize_fn=self._normalize_fn,
                )
            )

    def forward(self, inputs, num_frames):
        for block in self.blocks:
            inputs = block(inputs, num_frames=num_frames)
        return inputs
    
class TSMResNetV2(nn.Module):
    def __init__(
        self,
        normalize_fn = 'batchnorm',
        depth = 18,
        channel_shift_fraction = 0.125,
        width_mult = 1,
        output_stride = 8,
    ):
        super().__init__()

        if isinstance(channel_shift_fraction, float):
            channel_shift_fraction = [channel_shift_fraction] * 4

        if not all([0. <= x <= 1.0 for x in channel_shift_fraction]):
            raise ValueError(f'channel_shift_fraction ({channel_shift_fraction})'
                            ' all have to be in [0, 1].')

        self._output_stride= output_stride

        self._channels = (256, 512, 1024, 2048)

        if output_stride == 8:
            self.strides = (1, 2, 1, 1)
            self.rates = (1, 1, 2, 4)
        elif output_stride == 16:
            self.strides = (1, 2, 2, 1)
            self.rates = (1, 1, 1, 2)
        elif output_stride == 32:
            self.strides = (1, 2, 2, 2)
            self.rates = (1, 1, 1, 1)
        else:
            raise ValueError('unsupported output stride')

        num_blocks = {
            18: (2, 2, 2, 2),
            34: (3, 4, 6, 3),
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3),
            200: (3, 24, 36, 3),
        }
        if depth not in num_blocks:
            raise ValueError(
                f'`depth` should be in {list(num_blocks.keys())} ({depth} given).')
        self._num_blocks = num_blocks[depth]

        self._width_mult = width_mult
        self._channel_shift_fraction = channel_shift_fraction
        self._normalize_fn = normalize_fn
        self._use_bottleneck = (depth >= 50)

        self.conv = nn.Sequential(
            Conv2dSamePadding(3, 64 * self._width_mult, kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        in_ch = self._channels[0] * self._width_mult

        self.blocks = nn.ModuleList()
        for unit_id, (channels, num_blocks, stride, rate) in enumerate(
            zip(self._channels, self._num_blocks, self.strides, self.rates)):
            self.blocks.append(
                TSMResNetUnit(
                    in_ch,
                    channels * self._width_mult,
                    num_blocks=num_blocks,
                    stride=stride,
                    rate=rate,
                    channel_shift_fraction=self._channel_shift_fraction[unit_id],
                    tsm_mode='gpu',
                    use_bottleneck=self._use_bottleneck,
                    normalize_fn=self._normalize_fn,
                )
            )
            in_ch = channels * self._width_mult
        
        norm_channel = self._channels[-1] if self._use_bottleneck else self._channels[-1] // 4
        if normalize_fn == 'batchnorm':
            self.norm_0 = nn.BatchNorm2d(norm_channel)
        elif normalize_fn == 'groupnorm':
            self.norm_0 = nn.GroupNorm(norm_channel // 16, norm_channel)

    def forward(
        self,
        inputs,
        final_endpoint = 'Embeddings',
    ):
        """
            inputs: Tensor shape of [batch_size, channels, num_frames, height, width]
        """
        num_frames = inputs.shape[2]
        inputs = rearrange(inputs, 'b c t h w -> (b t) c h w')

        self._final_endpoint = final_endpoint

        end_point = 'tsm_resnet_stem'
        net = self.conv(inputs)
        if self._final_endpoint == end_point:
            net = prepare_output(net, num_frames, reduce_mean=False)
            return net

        for unit_id, block in enumerate(self.blocks):
            end_point = f'tsm_resnet_unit_{unit_id}'
            net = block(net, num_frames=num_frames)
            if self._final_endpoint == end_point:
                net = prepare_output(net, num_frames, reduce_mean=False)
                return net
        
        net = self.norm_0(net)
        net = F.relu(net)

        end_point = 'last_conv'
        if self._final_endpoint == end_point:
            net = prepare_output(net, num_frames, reduce_mean=False)
            return net

        net = net.mean(dim=(-1, -2))
        assert self._final_endpoint == 'Embeddings'
        return net

def test():
    model = TSMResNetV2(
        num_frames=24,
        channel_shift_fraction=[0.125, 0.125, 0., 0.],
    )
    model.eval()
    inputs = torch.randn(1, 3, 24, 256, 256)
    outputs = model(inputs, final_endpoint='tsm_resnet_unit_2')
    print(outputs.shape)

if __name__ == '__main__':
    test()

        