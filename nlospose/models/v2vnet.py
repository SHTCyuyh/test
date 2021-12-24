import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )
    
    def forward(self, x):  # [1, 32, 64, 64, 64]
        res = self.res_branch(x) # res = [1,32,64,64,64]; x = [1, 32, 64, 64, 64]
        skip = self.skip_con(x)  # [1, 32, 64, 64, 64]
        return F.relu(res + skip, True)

    
class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)
    

class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
    

class EncoderDecorder(nn.Module):
    def __init__(self):
        super(EncoderDecorder, self).__init__()

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)

        self.mid_res = Res3DBlock(128, 128)

        self.decoder_res2 = Res3DBlock(128, 128)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlock(64, 64)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2)

        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(64, 64)

    def forward(self, x):
        skip_x1 = self.skip_res1(x) # skip_x1 = [1,32,64,64,64]
        x = self.encoder_pool1(x) # [1, 32, 32, 32, 32]
        x = self.encoder_res1(x)  # [1, 64, 32, 32, 32]
        skip_x2 = self.skip_res2(x) # skip_x2 = [1, 64, 32, 32, 32]
        x = self.encoder_pool2(x) # [1, 64, 16, 16, 16]
        x = self.encoder_res2(x) # [1, 128, 16, 16, 16]

        x = self.mid_res(x) # [1, 128, 16, 16, 16]

        x = self.decoder_res2(x) # [1, 128, 16, 16, 16]
        x = self.decoder_upsample2(x) # [1, 64, 32, 32, 32]
        x = x + skip_x2 # [1, 64, 32, 32, 32]
        x = self.decoder_res1(x) # [1, 64, 32, 32, 32]
        x = self.decoder_upsample1(x) # [1, 32, 64, 64, 64]
        x = x + skip_x1 # [1, 32, 64, 64, 64]

        return x


class V2VModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(V2VModel, self).__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 16, 7),
            Pool3DBlock(2),
            Res3DBlock(16, 32),
            Res3DBlock(32, 32),
            Res3DBlock(32, 32)
        )

        self.encoder_decoder = EncoderDecorder()

        self.back_layers = nn.Sequential(
            Res3DBlock(32, 32),
            Basic3DBlock(32, 32, 1),
            Basic3DBlock(32, 32, 1),
        )

        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):  # (b, 1, 128, 128, 128)
        x = self.front_layers(x) # [1, 32, 64, 64, 64]
        x = self.encoder_decoder(x) # [1, 32, 64, 64, 64]
        x = self.back_layers(x) # [1, 32, 64, 64, 64]
        x = self.output_layer(x) # [1, 17, 64, 64, 64]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = V2VModel(1, 17).to("cuda")
    input = torch.ones((2,1,256,256,256)).to("cuda")
    output = model(input)
    # print(summary(model, input_size=(1,128,128,128), batch_size=2))
    print("f")