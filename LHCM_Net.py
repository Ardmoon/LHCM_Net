import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from SPPs import Faster_RSU4F,Faster_RSU6
from FasterNet import FasterNet

class GFFM(nn.Module):
    def __init__(self,c1=64,c2=128,c3=256,c4=512):
        super(GFFM, self).__init__()

        self.linear_c4 = nn.Conv2d(c4, 32, 1)
        self.linear_c3 = nn.Conv2d(c3, 32, 1)
        self.linear_c2 = nn.Conv2d(c2, 32, 1)
        self.linear_c1 = nn.Conv2d(c1, 32, 1)

        self.up_linear_fuse = Faster_RSU4F(128, 64)

        self.gated1 = nn.Sequential(nn.Conv2d(128,64,kernel_size=3,stride=2,padding=1,bias=False),nn.BatchNorm2d(64),nn.GELU(),
                                    nn.Conv2d(64, 64, kernel_size=1, stride=1),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=2,padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
                                    nn.Conv2d(64, 128, kernel_size=1, stride=1),
                                    torch.nn.AdaptiveMaxPool2d((1,1), return_indices=False),
                                    torch.nn.Sigmoid()
                                    )

        # self.linear_fuse = nn.Conv2d(128,64,kernel_size=1,stride=1,padding=0)
        self.outconv = nn.Conv2d(64, 128, 2, stride=2, padding=0)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.down8 = nn.AvgPool2d(kernel_size=8, stride=8)

    def forward(self, c1, c2, c3, c4):
        _c4 = self.linear_c4(c4)
        up_c4 = self.up8(_c4)
        down_c4 = _c4

        _c3 = self.linear_c3(c3)
        up_c3 = self.up4(_c3)
        down_c3 = self.down2(_c3)

        _c2 = self.linear_c2(c2)
        up_c2 = self.up2(_c2)
        down_c2 = self.down4(_c2)

        _c1 = self.linear_c1(c1)
        up_c1 = _c1
        down_c1 = self.down8(_c1)

        x = torch.cat([up_c4, up_c3, up_c2, up_c1], dim=1)   # 上采样

        gate = self.gated1(torch.cat([down_c4, down_c3, down_c2, down_c1], dim=1)) # 下采样后门控
        # print(gate.shape)
        # print(x.shape)

        x = x + x * gate

        x = self.outconv(self.up_linear_fuse(x))

        return x


def get_topk(x, k=10, dim=-3):
    # b, c, h, w = x.shape
    val, _ = torch.topk(x, k=k, dim=dim)
    return val


class ZeroWindow: # 给输入加权，削弱特征图中每个像素点自身及附近范围对自身的相关性。根据高斯分布削弱
    def __init__(self):
        self.store = {}

    def __call__(self, x_in, h, w, rat_s=0.1): #x_in是相似度矩阵,被reshape成了(b, w*h, h, w),h和w是相似度矩阵前面的特征图的高宽
        sigma = h * rat_s, w * rat_s
        # c = h * w
        b, c, h2, w2 = x_in.shape  # b, w*h, h, w
        key = str(x_in.shape) + str(rat_s)
        if key not in self.store:
            ind_r = torch.arange(h2).float()  #[0,1,2,3,...,h-1]
            ind_c = torch.arange(w2).float()  #[0,1,2,3,...,w-1]
            ind_r = ind_r.view(1, 1, -1, 1).expand_as(x_in)  #维度扩张+广播,(h) -> (1,1,h,1) -> (b, w*h, h, w)
            ind_c = ind_c.view(1, 1, 1, -1).expand_as(x_in)  #维度扩张+广播,(w) -> (1,1,1,w) -> (b, w*h, h, w)
            # ind_r和ind_c可以联合表示格子的位置

            # center
            c_indices = torch.from_numpy(np.indices((h, w))).float()   #shape=(2,h,w),可以表示格子的位置
            c_ind_r = c_indices[0].reshape(-1)  #[0,0,0,...,0,   1,1,1,...,1,   2,2,2,...]   #shape=(w*h)
            c_ind_c = c_indices[1].reshape(-1)  #[0,1,2,...,w-1, 0,1,2,...,w-1, 0,1,2,...]   #shape=(w*h)

            cent_r = c_ind_r.reshape(1, c, 1, 1).expand_as(x_in)  #(w*h) -> (1,w*h,1,1) -> (b, w*h, h, w)
            cent_c = c_ind_c.reshape(1, c, 1, 1).expand_as(x_in)  #(w*h) -> (1,w*h,1,1) -> (b, w*h, h, w)

            def fn_gauss(x, u, s):  # 高斯分布函数
                return torch.exp(-(x - u) ** 2 / (2 * s ** 2))

            gaus_r = fn_gauss(ind_r, cent_r, sigma[0]) #(b, w*h, h, w)
            gaus_c = fn_gauss(ind_c, cent_c, sigma[1]) #(b, w*h, h, w)
            out_g = 1 - gaus_r * gaus_c
            out_g = out_g.to(x_in.device)
            self.store[key] = out_g  # 把新值添加进字典
        else:
            out_g = self.store[key]
        out = out_g * x_in   # 加权，消除自己对自己的相关性。
        return out


class LCSCC(nn.Module):
    def __init__(self, topk=3,patch_size=2):
        super().__init__()
        # self.h = hw[0]
        # self.w = hw[1]
        self.topk = topk
        self.patch_size = patch_size
        self.zero_window = ZeroWindow()

        self.alpha = nn.Parameter(torch.tensor(5., dtype=torch.float32))

    def forward(self, x):  # x (b,c,h,w)
        patch_size = self.patch_size
        b,c,h,w = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        num_patch_h = int(h/patch_size)
        num_patch_w = int(w/patch_size)
        num_patches = num_patch_w*num_patch_h
        patch_area = patch_size*patch_size   # 一个patch包含几个像素点
        assert num_patches == (h*w)/patch_area


        x = F.normalize(x, p=2, dim=-3)  # 在通道维度归一化


        # 下面的N指patch个数num_patches，P指一个patch包含几个像素点patch_area
        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(b * c * num_patch_h, patch_size, num_patch_w, patch_size)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(b, c, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(b * patch_area, num_patches, -1)


        x = torch.matmul(x, x.transpose(1, 2))  # 相似度矩阵[BP, N, N]


        # zero out same area corr
        x = self.zero_window(x.view(b*patch_area, -1, num_patch_h, num_patch_w), num_patch_h, num_patch_w, rat_s=0.05).reshape(b*patch_area, num_patch_h*num_patch_w, num_patch_h*num_patch_w)
        # 把相似度矩阵reshape成(b,w*h,h,w),送入zero window加权，去除了自身与自身的相关性，再reshape回到(b, h1 * w1, h2 * w2)

        x = F.softmax(x * self.alpha, dim=-1) * F.softmax(x * self.alpha, dim=-2)   #归一化 （BP,N,N）

        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(b, patch_area, num_patches, -1)
        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(b * num_patches * num_patch_h, num_patch_w, patch_size, patch_size)
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(b, num_patches, num_patch_h * patch_size, num_patch_w * patch_size)

        x = get_topk(x, k=self.topk, dim=-3)  #在h1*w1维度选取top_k


        return x   #(b, top_k, h, w)


class atrous_spatial_pyramid_pooling_GELU(nn.Module):
    def __init__(self, in_channel, depth=128, rate_dict=[2, 4, 8, 12]):
        super(atrous_spatial_pyramid_pooling_GELU, self).__init__()

        self.modules = []
        for index, n_rate in enumerate(rate_dict):
            self.modules.append(nn.Sequential(
                nn.Conv2d(in_channel, depth, 3, dilation=n_rate, padding=n_rate, bias=False),
                nn.BatchNorm2d(depth),
                nn.GELU(),
                nn.Conv2d(depth, int(depth / 4), 1, dilation=n_rate, bias=False),
            ))

        self.convs = nn.ModuleList(self.modules)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        return torch.cat(res, 1)

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src





# input size = (b,3,320,320)
class LHCM_Net_320(nn.Module):
    def __init__(self):
        super(LHCM_Net_320, self).__init__()

        self.backbone = FasterNet()

        self.head = GFFM(c1=40,c2=80,c3=160,c4=160)
        self.corr1 = LCSCC(topk=32)  # 原 my_mobile_Corr(topk=32)


        self.side1 = nn.Conv2d(32, 1, 3, padding=1)  #  in_channe 原32， 只有cross corr用64， 记得改
        self.side2 = nn.Conv2d(64, 1, 3, padding=1)
        self.side3 = nn.Conv2d(32, 1, 3, padding=1)

        # in_channe 原32， 只有cross corr用64， 记得改
        self.aspp = nn.Sequential(nn.LayerNorm([32,80,80]),
            atrous_spatial_pyramid_pooling_GELU(in_channel=32, depth=64, rate_dict=[2, 4, 8, 12,16,20]),
            nn.Conv2d(96, 64, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.rsu = Faster_RSU6(64,32)

        self.outconv = nn.Conv2d(4, 1, 1)


    def forward(self, x):
        x0 = x
        x1, x2, x3, x4 = self.backbone.forward_feature(x)

        d0 = self.head(x1, x2, x3, x4)   # (b,128,80,80)
        d0 = self.corr1(d0)    # (b,32,80,80)

        s1 = self.aspp(d0)
        s2 = self.rsu(s1)

        d2 = _upsample_like(self.side2(s1), x0)

        d3 = _upsample_like(self.side3(s2), x0)


        return d3,d2


# input size = (b,3,256,256)
class LHCM_Net_256(nn.Module):
    def __init__(self):
        super(LHCM_Net_256, self).__init__()

        self.backbone = FasterNet()

        self.head = GFFM(c1=40,c2=80,c3=160,c4=160)

        self.corr1 = LCSCC(topk=32)  # 原 my_mobile_Corr(topk=32)

        self.side1 = nn.Conv2d(32, 1, 3, padding=1)  #  in_channe 原32， 只有cross corr用64， 记得改
        self.side2 = nn.Conv2d(64, 1, 3, padding=1)
        self.side3 = nn.Conv2d(32, 1, 3, padding=1)

        # in_channe 原32， 只有cross corr用64， 记得改
        self.aspp = nn.Sequential(nn.LayerNorm([32,64,64]),
            atrous_spatial_pyramid_pooling_GELU(in_channel=32, depth=64, rate_dict=[2, 4, 8, 12,16,20]),
            nn.Conv2d(96, 64, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.rsu = Faster_RSU6(64,32)

        self.outconv = nn.Conv2d(4, 1, 1)


    def forward(self, x):
        x0 = x
        x1, x2, x3, x4 = self.backbone.forward_feature(x)

        d0 = self.head(x1, x2, x3, x4)   # (b,128,80,80)
        d0 = self.corr1(d0)    # (b,32,80,80)

        s1 = self.aspp(d0)
        s2 = self.rsu(s1)


        d2 = _upsample_like(self.side2(s1), x0)

        d3 = _upsample_like(self.side3(s2), x0)


        return d3,d2






# if __name__ == '__main__':
#     device = torch.device('cpu')
#     x = torch.randn([16, 3, 320, 320]).to(device)
#
#     net = LHCM_Net_320().to(device)
#
#     a= net(x)
#     print(a[0].shape)
