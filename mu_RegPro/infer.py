import os

from monai.metrics import compute_dice

import utils
from torch.utils.data import DataLoader

from utils import print_metric, update_metric,compute_mTRE
import pandas as pd
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from model import Net
from utils import AverageMeter
import random


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


same_seeds(24)

def main():
    test_dir = '/home/gyl/DataSets/muProReg_process/val'  # 修改为 Biopsy 数据集的测试数据路径

    dict=["Dice","HD95","ASSD","TRE"]
    model_idx = -1
    lr = 0.0001
    weights = [0.01, 1, 1]  # loss weights
    model_folder = 'CROSS_JRS_mse_{}_weakly_{}_diffusion_{}_lr_{}/'.format(weights[0], weights[1],weights[2],lr)
    model_dir = 'experiments/' + model_folder
    if 'test' in test_dir:
        csv_name = model_folder[:-1]+'test'
    else:
        csv_name = model_folder[:-1]
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+csv_name+'.csv'):
        os.remove('Quantitative_Results/'+csv_name+'.csv')
    csv_writter(model_folder[:-1], 'Quantitative_Results/' + csv_name)
    line = ''
    for i in range(len(dict)):
        line = line + ',' + dict[i]
    csv_writter(line +','+'non_jec', 'Quantitative_Results/' + csv_name)
    # 定义保存文件
    # save_path = "/home/gyl/project/SegMorph/Biopsy/PGLFFNet/muProReg_all_methods_results.csv"

    # 假设当前方法名
    method_name = "CROSS-JRS"
    results_df=pd.DataFrame(columns=['Sample', 'Method', 'Dice'])

    img_size = (80, 80, 80)

    model = Net(img_size)
    # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx], map_location=device)['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.muProRegInferDataset(test_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = AverageMeter()
    eval_dsc_segx=AverageMeter()
    eval_hd95_def=AverageMeter()
    eval_assd_def=AverageMeter()
    eval_tre_def=AverageMeter()

    eval_dsc_raw = AverageMeter()
    eval_dsc_segy=AverageMeter()
    eval_hd95_raw=AverageMeter()
    eval_assd_raw=AverageMeter()
    eval_tre_raw=AverageMeter()
    eval_det = AverageMeter()
    metrics=eval_dsc_def,eval_dsc_raw,eval_hd95_def,eval_hd95_raw,eval_assd_def,eval_assd_raw,eval_tre_def,eval_tre_raw\
        ,eval_det
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2][:, :, :, :, :, 0].float()
            y_seg = data[3][:, :, :, :, :, 0].float()
            warped_mov,flow, segx,segy= model(x,y)


            def_seg = reg_model([x_seg.float(), flow.float()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            dsc_trans = compute_dice(def_seg.long(), y_seg.long())
            # 将预测转换为类别
            segx = torch.sigmoid(segx)
            segx = (segx > 0.5).float()
            segy = torch.sigmoid(segy)
            segy = (segy > 0.5).float()

            # 计算分割的dice
            dsc_x=compute_dice(segx.long(), x_seg.long())
            eval_dsc_segx.update(dsc_x.item())
            dsc_y=compute_dice(segy.long(), y_seg.long())
            eval_dsc_segy.update(dsc_y.item())

            hd95_def, assd_def = utils.compute_seg_metric(def_seg.long(), y_seg.long())
            hd95_raw, assd_raw = utils.compute_seg_metric(x_seg.long(), y_seg.long())

            def_segs=[]
            for i in range(6):
                def_seg_i=reg_model([data[2][...,i].float(),flow.float()])
                def_segs.append(def_seg_i)
            def_segs=torch.stack(def_segs,dim=-1)
            tre_def=compute_mTRE(def_segs.float(),data[3].float())
            tre_raw=compute_mTRE(data[2].float(),data[3].float())

            line = str(dsc_trans.item())
            line = line + ',' + str(hd95_def) + ',' + str(assd_def) + ',' + str(tre_def)
            line = line + ',' + str(np.sum(jac_det <= 0) / np.prod(tar.shape))
            csv_writter(line, 'Quantitative_Results/' + model_folder[:-1])

            dsc_raw = compute_dice(x_seg.long(), y_seg.long())

            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            metric=dsc_trans, dsc_raw, hd95_def, hd95_raw, assd_def, assd_raw, tre_def, tre_raw, jac_det
            update_metric(metrics,metric,x,tar)

            # 保存量化结果
            results_df.loc[len(results_df)] = [stdy_idx, method_name, dsc_trans.item()]
            stdy_idx += 1
        # 保存到CSV
        # if not os.path.exists(save_path):
        #     results_df.to_csv(save_path, index=False)
        # else:
        #     results_df.to_csv(save_path, mode='a', header=False, index=False)
    print('SegX DSC:{:.3f} +- {:.3f}, SqgY DSC:{:.3f} +- {:.3f}'.format(eval_dsc_segx.avg,
                                                                        eval_dsc_segx.std,
                                                                        eval_dsc_segy.avg,
                                                                        eval_dsc_segy.std))
    print_metric(metrics)


def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()