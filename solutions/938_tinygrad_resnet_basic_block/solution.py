from tinygrad import Tensor, nn

class ResidualBlock:
    def __init__(self, 
    in_channels, 
    out_channels, 
    stride=1):
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.uses_projection = True if in_channels != out_channels or stride != 1 else False 

        if self.uses_projection:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def __call__(self, x):
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out)).relu()

        shortcut = self.shortcut_bn(self.shortcut_conv(x)) if self.uses_projection else x
    
        return (out + shortcut).relu()

