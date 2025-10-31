import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F
from torch.distributions import Normal


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class Reg_Head(nn.Module):
    def __init__(self,in_ch):
        super().__init__()
        self.defconv = nn.Conv3d(in_ch, 3, 3, 1, 1)
        self.defconv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv.weight.shape))
        self.defconv.bias = nn.Parameter(torch.zeros(self.defconv.bias.shape))

    def forward(self, x):
        flow=self.defconv(x)
        return flow

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d((2*in_channels)//3, (2*in_channels)//3, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 处理尺寸不匹配
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)

class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=16):
        super(Encoder, self).__init__()

        c = first_out_channel
        self.conv0 = ConvInsBlock(in_channel, c, 3, 1)

        self.conv1 = nn.Sequential(
            nn.Conv3d(c, 2 * c, kernel_size=3, stride=2, padding=1),  # 80
            ResBlock(2 * c)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1),  # 40
            ResBlock(4 * c)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(4 * c, 8 * c, kernel_size=3, stride=2, padding=1),  # 20
            ResBlock(8 * c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8

        return [out0, out1, out2, out3]
class CConv(nn.Module):
    def __init__(self, channel):
        super(CConv, self).__init__()

        c = channel

        self.conv = nn.Sequential(
            ConvInsBlock(c, c, 3, 1),
            ConvInsBlock(c, c, 3, 1)
        )

    def forward(self, float_fm, fixed_fm, d_fm):
        concat_fm = torch.cat([float_fm, fixed_fm, d_fm], dim=1)
        x = self.conv(concat_fm)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction):
        super(SEBlock, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool3d(1)
        self.fc=nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,h,w,d = x.shape
        y=self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1,1)
        return x*y.expand_as(x)

class SegEncoder(nn.Module):
    def __init__(self, n_channels,base_channels=16,p_drop=0.1):
        super(SegEncoder, self).__init__()
        self.n_channels = n_channels

        c=base_channels
        self.channels = [c, 2*c, 4*c, 8*c, 16*c]

        # 初始卷积
        self.inc = DoubleConv(n_channels, self.channels[0])

        # 编码器
        self.down1 = Down(self.channels[0], self.channels[1])
        self.down2 = Down(self.channels[1], self.channels[2])
        self.down3 = Down(self.channels[2], self.channels[3])
        self.down4 = Down(self.channels[3], self.channels[4])

        self.drop2 = nn.Dropout3d(p=p_drop)      # 中层
        self.drop3 = nn.Dropout3d(p=p_drop)      # 中层
        self.drop4 = nn.Dropout3d(p=p_drop)      # 中层
        self.drop5 = nn.Dropout3d(p=min(p_drop*2, 0.5))  # 瓶颈层


    def forward(self, x):
        # 编码器部分
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2=self.drop2(x2)
        x3 = self.down2(x2)
        x3=self.drop3(x3)
        x4 = self.down3(x3)
        x4=self.drop4(x4)
        x5 = self.down4(x4)
        x5=self.drop5(x5)

        enc=[x1,x2,x3,x4,x5]


        return enc

class Decoder(nn.Module):
    def __init__(self, c=16,bilinear=True):
        super(Decoder, self).__init__()
        self.channels = [c, 2 * c, 4 * c, 8 * c, 16 * c]


        # 解码器 - 与编码器通道数完全对称
        # 解码器 - 关键是这里的输入通道数必须考虑拼接
        # 实际输入通道数为：当前层通道数 + 对应编码器层的通道数
        self.up1 = Up(self.channels[4] + self.channels[3], self.channels[3], bilinear)
        self.up2 = Up(self.channels[3] + self.channels[2], self.channels[2], bilinear)
        self.up3 = Up(self.channels[2] + self.channels[1], self.channels[1], bilinear)
        self.up4 = Up(self.channels[1] + self.channels[0], self.channels[0], bilinear)

    def forward(self,enc):
        x1,x2,x3,x4,x5 = enc
        dec_x4=self.up1(x5, x4)
        dec_x3=self.up2(dec_x4, x3)
        dec_x2=self.up3(dec_x3, x2)
        dec_x1=self.up4(dec_x2, x1)
        dec=[dec_x1,dec_x2,dec_x3,dec_x4]
        return dec

def kernel_size(c):
        k=int((math.log2(c)+1)//2)
        if k%2==0:
            return k+1
        else:
            return k

class Attn_Fusion_Gate(nn.Module):
    def __init__(self,in_channels):
        super(Attn_Fusion_Gate, self).__init__()
        self.in_channels = in_channels
        self.joint_distribution=nn.Sequential(
            DoubleConv(2 * in_channels, in_channels),
            SEBlock(in_channels,2),
        )
        self.dis_1=DoubleConv(2 * in_channels, in_channels)
        self.dis_2=DoubleConv(2 * in_channels, in_channels)
        self.attn=nn.Sequential(
            nn.Conv3d(2*in_channels,2,3,padding=1,groups=2),
            nn.Sigmoid(),
            nn.Conv3d(2,2,3,1,1),
            nn.Softmax(dim=1)
        )

    def forward(self,t1,t2):
        b,c,h,w,d = t1.shape
        x=torch.cat((t1,t2),dim=1)
        jonit=self.joint_distribution(x)
        t1=torch.cat((t1,jonit),dim=1)
        t2=torch.cat((t2,jonit),dim=1)
        t1=self.dis_1(t1)
        t2=self.dis_2(t2)
        x=torch.cat((t1,t2),dim=1)
        attn=self.attn(x).reshape(b,2,1,h,w,d).permute(1,0,2,3,4,5)
        fuse=t1*attn[0] + t2*attn[1]
        return fuse

class ESTBlock(nn.Module):
    def __init__(self,in_ch,inshape):
        super().__init__()
        self.conv_reg=nn.Sequential(
            ConvInsBlock(3*in_ch,in_ch),
            ConvInsBlock(in_ch,in_ch),
        )
        self.def_conv=Reg_Head(in_ch)
        self.conv_seg=nn.Sequential(
            ConvInsBlock(2*in_ch,in_ch),
            ConvInsBlock(in_ch,in_ch),

        )
        self.def_conv_seg=Reg_Head(in_ch)
        self.fuse_conv_dec = Attn_Fusion_Gate(in_ch)
        self.diff=VecInt(inshape)
        self.warp=SpatialTransformer(inshape)
        self.up_sample=nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True)
        self.UpConv=UpConvBlock(in_ch*2,in_ch)

    def forward(self,mov,fix,f_seg_m,f_seg_f,flow,reg_feat,f_seg_enc_m,f_seg_enc_f):
        flow=self.up_sample(2*flow)
        reg_feat=self.UpConv(reg_feat)
        warped_seg=self.warp(f_seg_m,flow)
        v_seg_feat=torch.cat([f_seg_f,warped_seg],dim=1)
        v_seg_feat=self.conv_seg(v_seg_feat)
        v_seg=self.def_conv_seg(v_seg_feat)
        flow_seg=self.diff(v_seg)
        flow=self.warp(flow,flow_seg)+flow_seg
        mov=self.fuse_conv_dec(mov,f_seg_m)
        fix=self.fuse_conv_dec(fix,f_seg_f)
        warped_reg=self.warp(mov,flow)
        v_reg_feat=torch.cat((fix,warped_reg,reg_feat),dim=1)
        v_reg_feat=self.conv_reg(v_reg_feat)
        v_reg=self.def_conv(v_reg_feat)
        flow_reg=self.diff(v_reg)
        flow_reg_inv=self.diff(-v_reg)

        flow=self.warp(flow,flow_reg)+flow_reg
        return flow,v_reg_feat


class RegPyramid(nn.Module):
    def __init__(self, inshape=(128,128,32),c=16):
        super(RegPyramid, self).__init__()
        self.channels = [c, 2 * c, 4 * c, 8 * c,16 * c]
        self.Init_def=nn.Sequential(
            ConvInsBlock(self.channels[4]*2,self.channels[4]),
        )
        self.init_def=Reg_Head(self.channels[4])
        self.diff=VecInt([s//2**4 for s in inshape])
        self.estimate=nn.ModuleList()
        for i in range(4):
            self.estimate.append(ESTBlock(self.channels[i],[s//2**i for s in inshape]))

    def forward(self,reg_ms,reg_fs,dec_ms,dec_fs,enc_ms,enc_fs):
        seg_dec=torch.cat([enc_ms[4],enc_fs[4]],dim=1)
        reg_feat=self.Init_def(seg_dec)
        v=self.init_def(reg_feat)
        flow=self.diff(v)
        # save_dir = f'./features/7_regEnhanceSeg_v4_dropout/{id}'
        # import os
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # from Biopsy.PGLFFNet.models.utils import save_deformation_field_nii
        # save_deformation_field_nii(flow,save_dir=save_dir,title='flow5')
        for i in range(4):

            flow,reg_feat=self.estimate[3-i](reg_ms[3-i],reg_fs[3-i],dec_ms[3-i],dec_fs[3-i],
                                             flow,reg_feat,enc_ms[3-i],enc_fs[3-i])
            # save_deformation_field_nii(flow,save_dir=save_dir,title=f'flow{4-i}')


        return flow

class CrossEnhanceBlock(nn.Module):
    def __init__(self,in_ch,inshape):
        super(CrossEnhanceBlock, self).__init__()
        self.conv=DoubleConv(2*in_ch,in_ch)
        self.def_conv=Reg_Head(in_ch)
        self.diff=VecInt(inshape)
        self.warp=SpatialTransformer(inshape)
        self.fuse_conv=Attn_Fusion_Gate(in_ch)
        self.seg_into_reg=DoubleConv(2*in_ch,in_ch)

    def forward(self,x,y,reg_enc_m,reg_enc_f):
        reg_enc_m=torch.cat([reg_enc_m,x],dim=1)
        reg_enc_f=torch.cat([reg_enc_f,y],dim=1)
        reg_enc_m=self.seg_into_reg(reg_enc_m)
        reg_enc_f=self.seg_into_reg(reg_enc_f)
        reg_feat=torch.cat([reg_enc_m,reg_enc_f],dim=1)
        reg_feat=self.conv(reg_feat)
        v=self.def_conv(reg_feat)
        flow_xy=self.diff(v)
        flow_yx=self.diff(-v)
        warped_x=self.warp(x,flow_xy)
        warped_y=self.warp(y,flow_yx)
        enhance_x=self.fuse_conv(x,warped_y)
        enhance_y=self.fuse_conv(y,warped_x)
        # save_dir = f'./features/7_regEnhanceSeg_v4_dropout/{PGLFFNet.id}'
        # import os
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # from Biopsy.PGLFFNet.models.utils import save_deformation_field_nii
        # save_deformation_field_nii(flow_xy, save_dir=save_dir, title='flow_xy')
        # save_deformation_field_nii(flow_yx, save_dir=save_dir, title='flow_yx')
        return enhance_x,enhance_y,reg_enc_m,reg_enc_f



class SegFusionNet(nn.Module):
    def __init__(self,inshape=(128,128,32),c=16,p_drop=0.05):
        super(SegFusionNet, self).__init__()
        self.channels = [c, 2 * c, 4 * c, 8 * c,16 * c]
        self.CEB_enc=nn.ModuleList()
        self.up=nn.ModuleList()
        for i in range(4):
            self.CEB_enc.append(CrossEnhanceBlock(self.channels[i],[s//2**i for s in inshape]))
        for i in range(4):
            self.up.append(Up(self.channels[i]+self.channels[i+1],self.channels[i],False))

        self.outc=OutConv(c,1)

        self.drop2 = nn.Dropout3d(p=p_drop*0.2)      # 中层
        self.drop3 = nn.Dropout3d(p=p_drop*0.2)      # 中层
        self.drop4 = nn.Dropout3d(p=p_drop*0.4)      # 中层

    def forward(self,seg_enc_m,seg_enc_f,reg_enc_m,reg_enc_f):
        dec_m4=self.up[3](seg_enc_m[4],seg_enc_m[3])
        dec_f4=self.up[3](seg_enc_f[4],seg_enc_f[3])
        dec_m4=self.drop4(dec_m4)
        dec_f4=self.drop4(dec_f4)

        dec_m4,dec_f4,reg_enc_m4,reg_enc_f4=self.CEB_enc[3](dec_m4,dec_f4,reg_enc_m[3],reg_enc_f[3])
        dec_m3=self.up[2](dec_m4,seg_enc_m[2])
        dec_f3=self.up[2](dec_f4,seg_enc_f[2])
        dec_m3=self.drop3(dec_m3)
        dec_f3=self.drop3(dec_f3)

        dec_m3,dec_f3,reg_enc_m3,reg_enc_f3=self.CEB_enc[2](dec_m3,dec_f3,reg_enc_m[2],reg_enc_f[2])
        dec_m2=self.up[1](dec_m3,seg_enc_m[1])
        dec_f2=self.up[1](dec_f3,seg_enc_f[1])
        dec_m2=self.drop2(dec_m2)
        dec_f2=self.drop2(dec_f2)

        dec_m2,dec_f2,reg_enc_m2,reg_enc_f2=self.CEB_enc[1](dec_m2,dec_f2,reg_enc_m[1],reg_enc_f[1])
        dec_m1=self.up[0](dec_m2,seg_enc_m[0])
        dec_f1=self.up[0](dec_f2,seg_enc_f[0])

        dec_m1,dec_f1,reg_enc_m1,reg_enc_f1=self.CEB_enc[0](dec_m1,dec_f1,reg_enc_m[0],reg_enc_f[0])
        logits_x=self.outc(dec_m1)
        logits_y=self.outc(dec_f1)

        dec_m=[dec_m1,dec_m2,dec_m3,dec_m4]
        dec_f=[dec_f1,dec_f2,dec_f3,dec_f4]

        reg_enc_m_enhance=[reg_enc_m1,reg_enc_m2,reg_enc_m3,reg_enc_m4]
        reg_enc_f_enhance=[reg_enc_f1,reg_enc_f2,reg_enc_f3,reg_enc_f4]
        return dec_m,dec_f,logits_x,logits_y,reg_enc_m_enhance,reg_enc_f_enhance


# 用双层卷积将所有的分割特征和配准特征的融合，并且将GLNet变为Pyramid GL
# 在7的基础进行修改，先用配准的特征去增强分割的decoder特征，再将分割的encoder和增强后的decoder和配准进行融合，
# 形成配准增强分割，分割增强配准
# 和v2不同的是，后面配准的特征用的是一开始两个encoder特征融合的特征
class Net(nn.Module):
    id=0
    def __init__(self, inshape=(128,128,32),in_channel=1,ch=16,bilinear=True):
        super(Net, self).__init__()
        self.SegEncoder=SegEncoder(in_channel,ch)
        self.SegEnhance=SegFusionNet(inshape,ch)
        self.RegEncoder=Encoder(in_channel, ch)
        c=ch
        self.channels=[ch,2*ch,4*ch,8*ch]

        self.warp = SpatialTransformer(inshape)

        # bottleNeck
        self.RegDecoder=RegPyramid(inshape,c)

    def forward(self,mov,fix):
        seg_enc_m=self.SegEncoder(mov)
        seg_enc_f=self.SegEncoder(fix)
        reg_enc_m=self.RegEncoder(mov)
        reg_enc_f=self.RegEncoder(fix)
        seg_dec_m,seg_dec_f,logits_x,logits_y,reg_enc_m,reg_enc_f=self.SegEnhance(seg_enc_m,seg_enc_f,reg_enc_m,reg_enc_f)
        # 可视化特征图
        # id=PGLFFNet.id
        # PGLFFNet.id+=1
        # save_dir = f'./features/18_segEnhanceSeg/{id}'
        # import os
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # from Biopsy.PGLFFNet.models.utils import visualize_cam_upsample
        # visualize_cam_upsample(seg_dec_m_old,title='seg_dec_m_old',save_dir=save_dir)
        # visualize_cam_upsample(seg_dec_m,title='seg_dec_m',save_dir=save_dir)
        # visualize_cam_upsample(seg_enc_m,title='seg_enc_m',save_dir=save_dir)




        flow=self.RegDecoder(reg_enc_m,reg_enc_f,seg_dec_m,seg_dec_f,seg_enc_m,seg_enc_f)

        moved_img=self.warp(mov,flow)
        return moved_img,flow,logits_x,logits_y

if __name__ == '__main__':
    size = (1, 1, 128, 128, 32)
    model = Net(size[2:]).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params:,}")
    # print(str(model))
    A = torch.ones(size).cuda()
    B = torch.ones(size).cuda()
    AL = torch.ones(size)
    BL = torch.ones(size)

    out, flow, segx,segy= model(A, B)
    print(out.shape, flow.shape)
