from monai.metrics import compute_dice
import  pandas as pd
import utils
from torch.utils.data import DataLoader

from utils import print_metric, update_metric
from data import datasets, trans
from torchvision import transforms
from natsort import natsorted
from model import Net
from utils import *
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
    test_dir = '/home/gyl/DataSets/Biopsy/test'  # 修改为 Biopsy 数据集的测试数据路径

    dict=["Dice","HD95","ASSD","TRE"]
    model_idx = -1
    lr = 0.0001
    weights = [0.01, 1, 1]  # loss weights
    model_folder = 'CROSS_JRS_ncc_{}_weakly_{}_diffusion_{}_lr_{}_sameseed_100epoch/'.format(weights[0], weights[1],weights[2],lr)
    model_dir = 'experiments/' + model_folder+'reg/'
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


    img_size=(128, 128, 32)
    model =Net(img_size)
    # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx], map_location=device)['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    best_epoch = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx], map_location=device)['epoch']
    print(f'Best epoch: {best_epoch}')
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.BiopsyInferDataset(test_dir, transforms=test_composed)
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
            x_seg = data[2]
            y_seg = data[3]
            warped_x,flow, segx,segy= model(x,y)

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

            hd95_def, assd_def, tre_def = utils.compute_seg_metric(def_seg.long(), y_seg.long())
            hd95_raw, assd_raw, tre_raw = utils.compute_seg_metric(x_seg.long(), y_seg.long())

            line = str(dsc_trans.item())
            line = line + ',' + str(hd95_def) + ',' + str(assd_def) + ',' + str(tre_def)
            line = line + ',' + str(np.sum(jac_det <= 0) / np.prod(tar.shape))
            csv_writter(line, 'Quantitative_Results/' + model_folder[:-1])

            dsc_raw = compute_dice(x_seg.long(), y_seg.long())

            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            metric=dsc_trans, dsc_raw, hd95_def, hd95_raw, assd_def, assd_raw, tre_def, tre_raw, jac_det
            update_metric(metrics,metric,x,tar)


            # 保存量化结果
            stdy_idx += 1

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
