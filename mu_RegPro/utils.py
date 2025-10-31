import math
import numpy as np
import torch.nn.functional as F
import torch
from monai.metrics import HausdorffDistanceMetric,compute_average_surface_distance
from natsort.compat.fake_fastnumbers import NAN_INF
from torch import nn
import pystrum.pynd.ndutils as nd
from scipy.ndimage import gaussian_filter

def read_txt_landmarks(filename, if_down=False):
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    landmarks = []
    for line in Lines:
        line = line.strip().split('\t')
        #print("Line: {}".format(line))
        if if_down:
            lm = [int(int(line[0])/2), int(int(line[1])/2), int(line[2])]
        else:
            lm = [int(line[0]), int(line[1]), int(line[2])]
        landmarks.append(lm)
    return landmarks

def deform_landmarks(source_lms, target_lms, flow, if_down=False):
    factor = 1
    if if_down:
        factor = 2
    u = flow[0, :, :, :]
    v = flow[1, :, :, :]
    w = flow[2, :, :, :]
    pixel_wth = [2.5, 0.97*factor, 0.97*factor]
    flow_fields = [u, v, w]
    diff_all = []
    raw_diff_all = []
    for i in range(len(source_lms)):
        source_lm = source_lms[i]
        target_lm = target_lms[i]
        diff = 0
        raw_diff = 0
        for j in range(len(source_lm)):
            sor_pnt = source_lm[2-j]
            tar_pnt = target_lm[2-j]
            def_field = flow_fields[j]
            out_pnt = sor_pnt - def_field[source_lm[2]-1, source_lm[1]-1, source_lm[0]-1]
            diff += (np.abs(out_pnt-tar_pnt)*pixel_wth[j])**2
            raw_diff += (np.abs(sor_pnt-tar_pnt)*pixel_wth[j])**2
        diff_all.append(math.sqrt(diff))
        raw_diff_all.append(math.sqrt(raw_diff))
    return np.mean(np.array(diff_all)), np.mean(np.std(diff_all)), np.mean(np.array(raw_diff_all)), np.std(np.array(raw_diff_all))

def dice_val(y_pred, y_true):
    intersection = (y_pred * y_true).sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2. * intersection) / (union + 1e-5)
    return torch.mean(dsc)

def dice_val_VOI(y_pred, y_true):
    VOI_lbls = [0]
    # print(y_pred.shape)
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred [:,:,:,i]
        true_i = true[:,:,:,i]
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2.*intersection) / (union + 1e-5)
        DSCs[idx] =dsc
        idx += 1
    return np.mean(DSCs)

def dice_val_substruct(y_pred, y_true, std_idx):
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=46)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=46)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)
    for i in range(1,2):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.*intersection) / (union + 1e-5)
        line = line+','+str(dsc)
    return line

def dice_val_CTsubstruct(y_pred, y_true, std_idx):
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=16)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=16)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)
    for i in range(16):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.*intersection) / (union + 1e-5)
        line = line+','+str(dsc)
    return line

import re
def process_label():
    seg_table = [0, 1,2]


    file1 = open('label_info.txt', 'r')
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ',line).split(' ')
            try:
                int(line[0])
            except:
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict

def process_CT_label():
    seg_table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    seg_name = ['Body-Outline', 'Bone-Structure', 'Right-Lung', 'Left-Lung', 'Heart', 'Liver', 'Spleen', 'Right-Kidney',
                'Left-Kidney', 'Stomach', 'Pancreas', 'Large-Intestine', 'Prostate', 'Bladder', 'Gall-Bladder', 'Thyroid']
    return seg_name

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[1:]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, 0)
    #print(grid)
    #sys.exit(0)

    # compute gradients
    [xFX, xFY, xFZ] = np.gradient(grid[0] - disp[0])
    [yFX, yFY, yFZ] = np.gradient(grid[1] - disp[1])
    [zFX, zFY, zFZ] = np.gradient(grid[2] - disp[2])

    jac_det = np.zeros(grid[0].shape)
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            for k in range(grid.shape[3]):
                jac_mij = [[xFX[i, j, k], xFY[i, j, k], xFZ[i, j, k]], [yFX[i, j, k], yFY[i, j, k], yFZ[i, j, k]], [zFX[i, j, k], zFY[i, j, k], zFZ[i, j, k]]]
                jac_det[i, j, k] =  np.linalg.det(jac_mij)

    # 3D glow
    #if nb_dims == 3:
    #    dx = J[0]
    #    dy = J[1]
    #    dz = J[2]

        # compute jacobian components
    #    Jdet0 = dx[0, ...] * (dy[1, ...] * dz[2, ...] - dy[2, ...] * dz[1, ...])
    #    Jdet1 = dx[1, ...] * (dy[0, ...] * dz[2, ...] - dy[2, ...] * dz[0, ...])
    #    Jdet2 = dx[2, ...] * (dy[0, ...] * dz[1, ...] - dy[1, ...] * dz[0, ...])

    #    return Jdet0 - Jdet1 + Jdet2

    #else:  # must be 2

    #    dfdx = J[0]
    #    dfdy = J[1]

    #    return dfdx[0, ...] * dfdy[1, ...] - dfdy[0, ...] * dfdx[1, ...]
    return jac_det

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def dice(y_pred, y_true, ):
    intersection = y_pred * y_true
    intersection = np.sum(intersection)
    union = np.sum(y_pred) + np.sum(y_true)
    dsc = (2.*intersection) / (union + 1e-5)
    return dsc

def smooth_seg(binary_img, sigma=1.5, thresh=0.5):
    binary_img = gaussian_filter(binary_img.astype(np.float32()), sigma=sigma)
    binary_img = binary_img > thresh
    return binary_img

def checkboard(shape, block_sz = 20):
    sz = 20
    xvalue = 64
    yvalue = 64
    A = np.zeros((sz, sz))
    B = np.ones((sz, sz))
    C = np.zeros((sz*xvalue, sz*yvalue))
    m = sz
    n = 0
    num = 2
    for i in range(xvalue):
        n1=0
        m1=sz
        for j in range(yvalue):
            if num % 2 == 0:
                C[n:m, n1:m1] = A
                num += 1
            else:
                C[n:m, n1:m1] = B
                num += 1
            n1 = n1 + sz
            m1 = m1 + sz
        if yvalue%2 == 0:
            num = num + 1
        n = n + sz
        m = m + sz

    C = C[0:shape[0], 0:shape[1]]
    return C

def compute_tre(y_true, y_pred):
    # 找到 y_true 和 y_pred 中前景质心
    def compute_centroid(mask):
        centroids = []
        for i in range(mask.shape[0]):
            indices = torch.nonzero(mask[i] > 0.5, as_tuple=False)
            if indices.shape[0] > 0:
                centroid = torch.mean(indices.float(), dim=0)
            else:
                centroid = torch.zeros(4, device=mask.device)
            centroids.append(centroid)
        return torch.stack(centroids, dim=0)[:,1:]

    y_true_centroids = compute_centroid(y_true)
    # print(y_true_centroids)
    y_pred_centroids = compute_centroid(y_pred)
    # print(y_pred_centroids)
    # print(y_true_centroids)
    # print(y_pred_centroids)
    # 计算欧几里得距离
    distances = torch.norm(y_true_centroids - y_pred_centroids, dim=1)

    # 返回平均TRE损失
    return distances.item()
import numpy as np
import torch


def get_index(lmark_truth, lmark_pred, number):

    index_t = torch.where(lmark_truth == number)
    index_p = torch.where(lmark_pred == number)

  #  print(f" number: {number} \n index_t: {index_t}  \n index_p: {index_p}")
    #
    if len(index_p[0]) < 1:
        if torch.mean(index_t[0].float()) > 41:
            ind_x = 90
        else:
            ind_x = - 10
        if torch.mean(index_t[1].float()) > 41:
            ind_y = 90
        else:
            ind_y = - 10
        if torch.mean(index_t[2].float()) > 41:
            ind_z = 90
        else:
            ind_z = - 10
        index_p = [torch.tensor([ind_x]), torch.tensor([ind_y]), torch.tensor([ind_z])]
    return index_t, index_p


#

def get_distance_lmark(lmark_truth, lmark_pred, device):
    landmark_tot_distance = []


    for landmark in range(int(torch.max(lmark_truth))):
        index_t, index_p = get_index(lmark_truth[0,0 :, :, :], lmark_pred[0,0 :, :, :], 1)

        if len(index_t) != len(index_p):
            continue
        print(index_t, index_p)
        diff = torch.stack(index_t)[1:4, :].float().mean(dim=1) - torch.stack(index_p)[1:4, :].float().mean(dim=1)

        landmark_tot_distance.append((diff ** 2).sum().sqrt().to(device))

    if len(landmark_tot_distance) == 0:
        return torch.tensor([0.0]).to(device)
    else:
        return torch.mean(torch.stack(landmark_tot_distance))


def compute_mTRE(y_true, y_pred):
    tres=[]
    for i in range(6):
        tres.append(compute_tre(y_true[...,i], y_pred[...,i]))
    return sum(tres)/len(tres)

def compute_seg_metric(y_pre,y_true):
    hd95 = HausdorffDistanceMetric(percentile=95.0)
    distance_hd95 = hd95(y_pre,y_true)

    distance_ASSD = compute_average_surface_distance(symmetric=True,y_pred=y_pre,y=y_true)

    hd95_max_value=30.0
    assd_max_value=30.0

    if torch.isnan(distance_hd95).any() or torch.isinf(distance_hd95).any() :
        distance_hd95 = torch.tensor(hd95_max_value)
    if torch.isnan(distance_ASSD).any() or torch.isinf(distance_ASSD).any() :
        distance_ASSD = torch.tensor(assd_max_value)
    # tre=get_distance_lmark(y_true,y_pre,y_pre.device)
    return distance_hd95.item(), distance_ASSD.item()


class Dense3DSpatialTransformer(nn.Module):
    def __init__(self):
        super(Dense3DSpatialTransformer, self).__init__()

    def forward(self, input1, input2):
        return self._transform(input1, input2[:, 0], input2[:, 1], input2[:, 2])

    def _transform(self, input1, dDepth, dHeight, dWidth):
        batchSize = dDepth.shape[0]
        dpt = dDepth.shape[1]
        hgt = dDepth.shape[2]
        wdt = dDepth.shape[3]

        D_mesh, H_mesh, W_mesh = self._meshgrid(dpt, hgt, wdt)
        D_mesh = D_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        D_upmesh = dDepth + D_mesh
        H_upmesh = dHeight + H_mesh
        W_upmesh = dWidth + W_mesh

        return self._interpolate(input1, D_upmesh, H_upmesh, W_upmesh)

    def _meshgrid(self, dpt, hgt, wdt):
        d_t = torch.linspace(0.0, dpt-1.0, dpt).unsqueeze_(1).unsqueeze_(1).expand(dpt, hgt, wdt).cuda()
        h_t = torch.matmul(torch.linspace(0.0, hgt-1.0, hgt).unsqueeze_(1), torch.ones((1,wdt))).cuda()
        h_t = h_t.unsqueeze_(0).expand(dpt, hgt, wdt)
        w_t = torch.matmul(torch.ones((hgt,1)), torch.linspace(0.0, wdt-1.0, wdt).unsqueeze_(1).transpose(1,0)).cuda()
        w_t = w_t.unsqueeze_(0).expand(dpt, hgt, wdt)
        return d_t, h_t, w_t

    def _interpolate(self, input, D_upmesh, H_upmesh, W_upmesh):
        nbatch = input.shape[0]
        nch    = input.shape[1]
        depth  = input.shape[2]
        height = input.shape[3]
        width  = input.shape[4]

        img = torch.zeros(nbatch, nch, depth+2,  height+2, width+2).cuda()
        img[:, :, 1:-1, 1:-1, 1:-1] = input

        imgDpt = img.shape[2]
        imgHgt = img.shape[3]
        imgWdt = img.shape[4]

        # D_upmesh, H_upmesh, W_upmesh = [D, H, W] -> [BDHW,]
        D_upmesh = D_upmesh.view(-1).float()+1.0  # (BDHW,)
        H_upmesh = H_upmesh.view(-1).float()+1.0  # (BDHW,)
        W_upmesh = W_upmesh.view(-1).float()+1.0  # (BDHW,)

        # D_upmesh, H_upmesh, W_upmesh -> Clamping
        df = torch.floor(D_upmesh).int()
        dc = df + 1
        hf = torch.floor(H_upmesh).int()
        hc = hf + 1
        wf = torch.floor(W_upmesh).int()
        wc = wf + 1

        df = torch.clamp(df, 0, imgDpt-1)  # (BDHW,)
        dc = torch.clamp(dc, 0, imgDpt-1)  # (BDHW,)
        hf = torch.clamp(hf, 0, imgHgt-1)  # (BDHW,)
        hc = torch.clamp(hc, 0, imgHgt-1)  # (BDHW,)
        wf = torch.clamp(wf, 0, imgWdt-1)  # (BDHW,)
        wc = torch.clamp(wc, 0, imgWdt-1)  # (BDHW,)

        # Find batch indexes
        rep = torch.ones([depth*height*width, ]).unsqueeze_(1).transpose(1, 0).cuda()
        bDHW = torch.matmul((torch.arange(0, nbatch).float()*imgDpt*imgHgt*imgWdt).unsqueeze_(1).cuda(), rep).view(-1).int()

        # Box updated indexes
        HW = imgHgt*imgWdt
        W = imgWdt
        # x: W, y: H, z: D
        idx_000 = bDHW + df*HW + hf*W + wf
        idx_100 = bDHW + dc*HW + hf*W + wf
        idx_010 = bDHW + df*HW + hc*W + wf
        idx_110 = bDHW + dc*HW + hc*W + wf
        idx_001 = bDHW + df*HW + hf*W + wc
        idx_101 = bDHW + dc*HW + hf*W + wc
        idx_011 = bDHW + df*HW + hc*W + wc
        idx_111 = bDHW + dc*HW + hc*W + wc

        # Box values
        img_flat = img.view(-1, nch).float()  # (BDHW,C) //// C=1

        val_000 = torch.index_select(img_flat, 0, idx_000.long())
        val_100 = torch.index_select(img_flat, 0, idx_100.long())
        val_010 = torch.index_select(img_flat, 0, idx_010.long())
        val_110 = torch.index_select(img_flat, 0, idx_110.long())
        val_001 = torch.index_select(img_flat, 0, idx_001.long())
        val_101 = torch.index_select(img_flat, 0, idx_101.long())
        val_011 = torch.index_select(img_flat, 0, idx_011.long())
        val_111 = torch.index_select(img_flat, 0, idx_111.long())

        dDepth  = dc.float() - D_upmesh
        dHeight = hc.float() - H_upmesh
        dWidth  = wc.float() - W_upmesh

        wgt_000 = (dWidth*dHeight*dDepth).unsqueeze_(1)
        wgt_100 = (dWidth * dHeight * (1-dDepth)).unsqueeze_(1)
        wgt_010 = (dWidth * (1-dHeight) * dDepth).unsqueeze_(1)
        wgt_110 = (dWidth * (1-dHeight) * (1-dDepth)).unsqueeze_(1)
        wgt_001 = ((1-dWidth) * dHeight * dDepth).unsqueeze_(1)
        wgt_101 = ((1-dWidth) * dHeight * (1-dDepth)).unsqueeze_(1)
        wgt_011 = ((1-dWidth) * (1-dHeight) * dDepth).unsqueeze_(1)
        wgt_111 = ((1-dWidth) * (1-dHeight) * (1-dDepth)).unsqueeze_(1)

        output = (val_000*wgt_000 + val_100*wgt_100 + val_010*wgt_010 + val_110*wgt_110 +
                  val_001 * wgt_001 + val_101 * wgt_101 + val_011 * wgt_011 + val_111 * wgt_111)
        output = output.view(nbatch, depth, height, width, nch).permute(0, 4, 1, 2, 3)  #B, C, D, H, W
        return output

# Spatial Transformer 3D Net #################################################
class Dense3DSpatialTransformerNN(nn.Module):
    def __init__(self):
        super(Dense3DSpatialTransformerNN, self).__init__()

    def forward(self, input1, input2):
        return self._transform(input1, input2[:, 0], input2[:, 1], input2[:, 2])

    def _transform(self, input1, dDepth, dHeight, dWidth):
        batchSize = dDepth.shape[0]
        dpt = dDepth.shape[1]
        hgt = dDepth.shape[2]
        wdt = dDepth.shape[3]

        D_mesh, H_mesh, W_mesh = self._meshgrid(dpt, hgt, wdt)
        D_mesh = D_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        D_upmesh = dDepth + D_mesh
        H_upmesh = dHeight + H_mesh
        W_upmesh = dWidth + W_mesh

        return self._interpolate(input1, D_upmesh, H_upmesh, W_upmesh)

    def _meshgrid(self, dpt, hgt, wdt):
        d_t = torch.linspace(0.0, dpt-1.0, dpt).unsqueeze_(1).unsqueeze_(1).expand(dpt, hgt, wdt).cuda()
        h_t = torch.matmul(torch.linspace(0.0, hgt-1.0, hgt).unsqueeze_(1), torch.ones((1,wdt))).cuda()
        h_t = h_t.unsqueeze_(0).expand(dpt, hgt, wdt)
        w_t = torch.matmul(torch.ones((hgt,1)), torch.linspace(0.0, wdt-1.0, wdt).unsqueeze_(1).transpose(1,0)).cuda()
        w_t = w_t.unsqueeze_(0).expand(dpt, hgt, wdt)
        return d_t, h_t, w_t

    def _interpolate(self, input, D_upmesh, H_upmesh, W_upmesh):
        nbatch = input.shape[0]
        nch    = input.shape[1]
        depth  = input.shape[2]
        height = input.shape[3]
        width  = input.shape[4]

        img = torch.zeros(nbatch, nch, depth+2,  height+2, width+2).cuda()
        img[:, :, 1:-1, 1:-1, 1:-1] = input

        imgDpt = img.shape[2]
        imgHgt = img.shape[3]
        imgWdt = img.shape[4]

        # D_upmesh, H_upmesh, W_upmesh = [D, H, W] -> [BDHW,]
        D_upmesh = D_upmesh.view(-1).float()+1.0  # (BDHW,)
        H_upmesh = H_upmesh.view(-1).float()+1.0  # (BDHW,)
        W_upmesh = W_upmesh.view(-1).float()+1.0  # (BDHW,)

        # D_upmesh, H_upmesh, W_upmesh -> Clamping
        df = torch.floor(D_upmesh).int()
        dc = df + 1
        hf = torch.floor(H_upmesh).int()
        hc = hf + 1
        wf = torch.floor(W_upmesh).int()
        wc = wf + 1

        df = torch.clamp(df, 0, imgDpt-1)  # (BDHW,)
        dc = torch.clamp(dc, 0, imgDpt-1)  # (BDHW,)
        hf = torch.clamp(hf, 0, imgHgt-1)  # (BDHW,)
        hc = torch.clamp(hc, 0, imgHgt-1)  # (BDHW,)
        wf = torch.clamp(wf, 0, imgWdt-1)  # (BDHW,)
        wc = torch.clamp(wc, 0, imgWdt-1)  # (BDHW,)

        # Find batch indexes
        rep = torch.ones([depth*height*width, ]).unsqueeze_(1).transpose(1, 0).cuda()
        bDHW = torch.matmul((torch.arange(0, nbatch).float()*imgDpt*imgHgt*imgWdt).unsqueeze_(1).cuda(), rep).view(-1).int()

        # Box updated indexes
        HW = imgHgt*imgWdt
        W = imgWdt
        # x: W, y: H, z: D
        idx_000 = bDHW + df*HW + hf*W + wf
        idx_100 = bDHW + dc*HW + hf*W + wf
        idx_010 = bDHW + df*HW + hc*W + wf
        idx_110 = bDHW + dc*HW + hc*W + wf
        idx_001 = bDHW + df*HW + hf*W + wc
        idx_101 = bDHW + dc*HW + hf*W + wc
        idx_011 = bDHW + df*HW + hc*W + wc
        idx_111 = bDHW + dc*HW + hc*W + wc

        # Box values
        img_flat = img.view(-1, nch).float()  # (BDHW,C) //// C=1

        val_000 = torch.index_select(img_flat, 0, idx_000.long())
        val_100 = torch.index_select(img_flat, 0, idx_100.long())
        val_010 = torch.index_select(img_flat, 0, idx_010.long())
        val_110 = torch.index_select(img_flat, 0, idx_110.long())
        val_001 = torch.index_select(img_flat, 0, idx_001.long())
        val_101 = torch.index_select(img_flat, 0, idx_101.long())
        val_011 = torch.index_select(img_flat, 0, idx_011.long())
        val_111 = torch.index_select(img_flat, 0, idx_111.long())

        dDepth  = torch.round(dc.float() - D_upmesh)
        dHeight = torch.round(hc.float() - H_upmesh)
        dWidth  = torch.round(wc.float() - W_upmesh)

        wgt_000 = (dWidth*dHeight*dDepth).unsqueeze_(1)
        wgt_100 = (dWidth * dHeight * (1-dDepth)).unsqueeze_(1)
        wgt_010 = (dWidth * (1-dHeight) * dDepth).unsqueeze_(1)
        wgt_110 = (dWidth * (1-dHeight) * (1-dDepth)).unsqueeze_(1)
        wgt_001 = ((1-dWidth) * dHeight * dDepth).unsqueeze_(1)
        wgt_101 = ((1-dWidth) * dHeight * (1-dDepth)).unsqueeze_(1)
        wgt_011 = ((1-dWidth) * (1-dHeight) * dDepth).unsqueeze_(1)
        wgt_111 = ((1-dWidth) * (1-dHeight) * (1-dDepth)).unsqueeze_(1)

        output = (val_000*wgt_000 + val_100*wgt_100 + val_010*wgt_010 + val_110*wgt_110 +
                  val_001 * wgt_001 + val_101 * wgt_101 + val_011 * wgt_011 + val_111 * wgt_111)
        output = output.view(nbatch, depth, height, width, nch).permute(0, 4, 1, 2, 3)  #B, C, D, H, W
        return output

class register_model(nn.Module):
    def __init__(self,  img_sz=None, mode='bilinear'):
        super(register_model, self).__init__()
        if mode=='bilinear':
            self.spatial_trans = Dense3DSpatialTransformer()
        else:
            self.spatial_trans = Dense3DSpatialTransformerNN()

    def forward(self, x):
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def update_metric(metrics,metric,x,tar):

    eval_dsc_def,eval_dsc_raw,eval_hd95_def,eval_hd95_raw,eval_assd_def,eval_assd_raw,eval_tre_def,eval_tre_raw\
        ,eval_det=metrics
    dsc_trans,dsc_raw,hd95_def,hd95_raw,assd_def,assd_raw,tre_def,tre_raw\
        ,jac_det=metric

    eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

    eval_dsc_def.update(dsc_trans.item(), x.size(0))
    eval_dsc_raw.update(dsc_raw.item(), x.size(0))
    eval_hd95_def.update(hd95_def, x.size(0))
    eval_hd95_raw.update(hd95_raw, x.size(0))
    eval_assd_def.update(assd_def, x.size(0))
    eval_assd_raw.update(assd_raw, x.size(0))
    eval_tre_def.update(tre_def, x.size(0))
    eval_tre_raw.update(tre_raw, x.size(0))

def print_metric(metrics):

    eval_dsc_def,eval_dsc_raw,eval_hd95_def,eval_hd95_raw,eval_assd_def,eval_assd_raw,eval_tre_def,eval_tre_raw\
        ,eval_det=metrics


    print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                eval_dsc_def.std,
                                                                                eval_dsc_raw.avg,
                                                                                eval_dsc_raw.std))
    print('Deformed HD95: {:.3f} +- {:.3f}, Affine HD95: {:.3f} +- {:.3f}'.format(eval_hd95_def.avg,
                                                                                  eval_hd95_def.std,
                                                                                  eval_hd95_raw.avg,
                                                                                  eval_hd95_raw.std))
    print('Deformed ASSD: {:.3f} +- {:.3f}, Affine ASSD: {:.3f} +- {:.3f}'.format(eval_assd_def.avg,
                                                                                  eval_assd_def.std,
                                                                                  eval_assd_raw.avg,
                                                                                  eval_assd_raw.std))
    print('Deformed TRE: {:.3f} +- {:.3f}, Affine TRE: {:.3f} +- {:.3f}'.format(eval_tre_def.avg,
                                                                                eval_tre_def.std,
                                                                                eval_tre_raw.avg,
                                                                                eval_tre_raw.std))

    print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))