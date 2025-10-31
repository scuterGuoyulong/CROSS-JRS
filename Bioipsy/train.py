import glob

from monai.losses import DiceLoss
from torch.utils.tensorboard import SummaryWriter
import os, losses
import utils
import sys
from torch.utils.data import DataLoader

from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import matplotlib.pyplot as plt
from natsort import natsorted
# from concise_model import PMorph
from model import  Net
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

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
def freeze_seg(model):
    for param in model.seg_decoder.parameters():
        param.requires_grad = False
    # for param in model.SegUS.parameters():
    #     param.requires_grad = False
def main():
    batch_size = 1
    train_dir = '/home/gyl/DataSets/Biopsy/train'
    val_dir = '/home/gyl/DataSets/Biopsy/valid'
    lr = 0.0001
    weights = [0.01, 1, 1]  # loss weights
    save_dir = 'CROSS_JRS_ncc_{}_weakly_{}_diffusion_{}_lr_{}_sameseed_100epoch/'.format(weights[0], weights[1],weights[2],lr)
    if not os.path.exists('experiments/' + save_dir+'reg/'):
        os.makedirs('experiments/' + save_dir+'reg/')
    if not os.path.exists('experiments/' + save_dir+'segX/'):
        os.makedirs('experiments/' + save_dir+'segX/')
    if not os.path.exists('experiments/' + save_dir+'segY/'):
        os.makedirs('experiments/' + save_dir+'segY/')
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    epoch_start = 0
    max_epoch = 100
    img_size = (128, 128, 32)

    cont_training = False
    freeze_segbran=False

    '''
    Initialize model
    '''
    model = Net(img_size)
    model.cuda()

    # 是否冻结分割的分支
    if freeze_segbran:
        freeze_seg(model)

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 25
        model_dir = 'experiments/'+save_dir+'reg/'
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.BiopsyDataset(train_dir, transforms=train_composed)
    val_set = datasets.BiopsyInferDataset(val_dir, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    criterions += [DiceLoss()]
    segloss=DiceLoss(sigmoid=True,include_background=False)
    best_dsc = 0
    best_dsc_segX=0
    best_dsc_segY=0
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_label=utils.AverageMeter()
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            output = model(x,y)
            def_x_seg = reg_model_bilin([x_seg, output[1]])
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                if n==2:
                    curr_loss = loss_function(def_x_seg.float(),y_seg.float())*weights[1]
                elif n==1:
                    curr_loss = loss_function(output[n], y) * weights[2]
                else:
                    curr_loss = (loss_function(output[n], y)) * weights[0]
                loss_vals.append(curr_loss)
                loss += curr_loss
            # seg loss
            seglossx=segloss(output[2].float(),x_seg.float())
            loss+=seglossx
            seglossy=segloss(output[3].float(),y_seg.float())
            loss+=seglossy

            loss_label.update(loss_vals[2].item(), y.numel())
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img {:.4f},  Label {:.6f}, Reg: {:.6f}, SegX: {:.6f},SegY: {:.6f}'.format(idx, len(train_loader),
                                                                                                loss.item(),
                                                                                                loss_vals[0].item(),
                                                                                                loss_vals[2].item(),
                                                                                                loss_vals[1].item(),
                                                                                                seglossx,seglossy))
        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        writer.add_scalar('Loss_label/train', loss_label.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        segx_dsc = utils.AverageMeter()
        segy_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                grid_img = mk_grid_img(8, 1, img_size)
                output = model(x,y)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                dsc = utils.dice_val(def_out.long(), y_seg.long())
                segx= torch.sigmoid(output[2])
                segy = torch.sigmoid(output[3])
                segx = (segx > 0.5).float()
                segy = (segy > 0.5).float()
                segx_dice=utils.dice_val(segx,x_seg.float())
                segy_dice=utils.dice_val(segy,y_seg.float())

                eval_dsc.update(dsc.item(), 1)
                segx_dsc.update(segx_dice.item(), 1)
                segy_dsc.update(segy_dice.item(), 1)
                print(eval_dsc.avg)
            best_dsc = max(eval_dsc.avg, best_dsc)
            best_dsc_segX = max(segx_dsc.avg, best_dsc_segX)
            best_dsc_segY = max(segy_dsc.avg, best_dsc_segY)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            }, save_dir='experiments/' + save_dir +'reg/', filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
            writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
            writer.add_scalar('DSC/segX', segx_dsc.avg, epoch)
            writer.add_scalar('DSC/segY', segy_dsc.avg, epoch)

            plt.switch_backend('agg')
            pred_fig = comput_fig(def_out)
            grid_fig = comput_fig(def_grid)
            x_fig = comput_fig(x_seg)
            tar_fig = comput_fig(y_seg)
            writer.add_figure('Grid', grid_fig, epoch)
            plt.close(grid_fig)
            writer.add_figure('input', x_fig, epoch)
            plt.close(x_fig)
            writer.add_figure('ground truth', tar_fig, epoch)
            plt.close(tar_fig)
            writer.add_figure('prediction', pred_fig, epoch)
            plt.close(pred_fig)

            # 可视化segmentation结果
            plt.switch_backend('agg')
            input_fig = comput_fig(x)
            label_fig = comput_fig(x_seg)
            pred_fig = comput_fig(output[2])

            writer.add_figure('InputX', input_fig, epoch)
            plt.close(input_fig)
            writer.add_figure('Ground TruthX', label_fig, epoch)
            plt.close(label_fig)
            writer.add_figure('PredictionX', pred_fig, epoch)
            plt.close(pred_fig)

            plt.switch_backend('agg')
            input_fig = comput_fig(y)
            label_fig = comput_fig(y_seg)
            pred_fig = comput_fig(output[3])

            writer.add_figure('InputY', input_fig, epoch)
            plt.close(input_fig)
            writer.add_figure('Ground TruthY', label_fig, epoch)
            plt.close(label_fig)
            writer.add_figure('PredictionY', pred_fig, epoch)
            plt.close(pred_fig)
            loss_all.reset()

    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(128,128,32)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

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