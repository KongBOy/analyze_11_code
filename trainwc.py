# code to train world coord regression from RGB Image
# models are saved in checkpoints-wc/

import grad_loss
from utils import show_wc_tnsboard, get_lr
from loaders import get_loader
from models import get_model
from tqdm import tqdm
from torchvision import utils
from torch.utils import data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys
import os
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use('Agg')


def write_log_file(log_file_name, losses, epoch, lrate, phase):
    with open(log_file_name, 'a') as f:
        f.write("\n{} LRate: {} Epoch: {} Loss: {} MSE: {} GradLoss: ".format(
            phase, lrate, epoch, losses[0], losses[1], losses[2]))


def train(args):
    # Setup Dataloader
    data_loader = get_loader('doc3dwc')
    data_path   = args.data_path
    t_loader    = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=True)
    v_loader    = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols))

    n_classes   = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=1, shuffle=True)
    valloader   = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=1)

    # Setup Model
    model = get_model(args.arch, n_classes, in_channels=3)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))  ### 原GPU版
    model.cuda()  ### GPU跑

    # Activation
    htan = nn.Hardtanh(0, 1.0)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate, weight_decay=5e-4, amsgrad=True)

    # LR Scheduler
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Losses
    MSE = nn.MSELoss()
    loss_fn = nn.L1Loss()
    gloss = grad_loss.Gradloss(window_size=5, padding=2)

    epoch_start = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            epoch_start = checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    # Log file:
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    # activation_dataset_lossparams_augmentations_trainstart
    experiment_name = 'htan_doc3d_l1grad_bghsaugk_scratch'
    log_file_name = os.path.join(args.logdir, experiment_name + '.txt')
    if os.path.isfile(log_file_name):
        log_file = open(log_file_name, 'a')
    else:
        log_file = open(log_file_name, 'w+')

    log_file.write('\n---------------  ' + experiment_name + '  ---------------\n')
    log_file.close()

    # Setup tensorboard for visualization
    if args.tboard:
        # save logs in runs/<experiment_name>
        writer = SummaryWriter(comment=experiment_name)

    best_val_mse = 99999.0
    global_step = 0

    for epoch in range(epoch_start, args.n_epoch):
        avg_loss = 0.0
        avg_l1loss = 0.0
        avg_gloss = 0.0
        train_mse = 0.0
        model.train()

        for i, (images, labels) in enumerate(tqdm(trainloader)):
            ### 我自己加的 debugcode
            # print("i~~", i)
            # import matplotlib.pyplot as plt
            # images_show = images.numpy().transpose(0, 2, 3, 1)
            # labels_show = labels.numpy().transpose(0, 2, 3, 1)
            # fig, ax = plt.subplots(nrows=1, ncols=2)
            # ax[0].imshow(images_show[0])
            # ax[1].imshow(labels_show[0])
            # plt.show()
            # print("images.shape", images.shape)
            # print("labels.shape", labels.shape)

            images = Variable(images.cuda())  ### 原GPU版
            labels = Variable(labels.cuda())  ### 原GPU版
            ### 我自己加的 debugcode
            # images = Variable(images)  ### 我改的 CPU版
            # labels = Variable(labels)  ### 我改的 CPU版

            optimizer.zero_grad()
            outputs = model(images)
            pred = htan(outputs)
            g_loss, _, _ = gloss(pred, labels)
            l1loss = loss_fn(pred, labels)
            loss = l1loss  # +(0.2*g_loss)
            avg_l1loss += float(l1loss)
            avg_gloss += float(g_loss)
            avg_loss += float(loss)
            train_mse += float(MSE(pred, labels).item())

            loss.backward()
            optimizer.step()
            global_step += 1

            if (i + 1) % 50 == 0:
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f" %
                      (epoch + 1, args.n_epoch, i + 1, len(trainloader), avg_loss / 50.0))
                avg_loss = 0.0

            if args.tboard and (i + 1) % 20 == 0:
                show_wc_tnsboard(global_step, writer, images, labels, pred, 8, 'Train Inputs', 'Train WCs', 'Train Pred. WCs')
                writer.add_scalar('WC: L1 Loss/train'  , avg_l1loss / (i + 1), global_step)
                writer.add_scalar('WC: Grad Loss/train', avg_gloss / (i + 1), global_step)
                # break  ### debug用

        train_mse = train_mse / len(trainloader)
        avg_l1loss = avg_l1loss / len(trainloader)
        avg_gloss = avg_gloss / len(trainloader)
        print("Training L1:%4f" % (avg_l1loss))
        print("Training MSE:'{}'".format(train_mse))
        train_losses = [avg_l1loss, train_mse, avg_gloss]

        lrate = get_lr(optimizer)
        write_log_file(experiment_name, train_losses, epoch + 1, lrate, 'Train')

        # model.eval()
        # val_loss = 0.0
        # val_mse = 0.0
        # val_bg = 0.0
        # val_fg = 0.0
        # val_gloss = 0.0
        # val_dloss = 0.0
        # for i_val, (images_val, labels_val) in enumerate(tqdm(valloader)):
        #     with torch.no_grad():
        #         images_val = Variable(images_val.cuda())
        #         labels_val = Variable(labels_val.cuda())

        #         outputs = model(images_val)
        #         pred_val = htan(outputs)
        #         g_loss, _, _ = gloss(pred_val, labels_val)
        #         # g_loss = gloss.cpu()
        #         pred_val = pred_val.cpu()
        #         labels_val = labels_val.cpu()
        #         loss = loss_fn(pred_val, labels_val)
        #         val_loss += float(loss)
        #         val_mse += float(MSE(pred_val, labels_val))
        #         val_gloss += float(g_loss)

        #         if(i_val + 1) % 5 == 0: break  ### 自己加的， 覺得不用看那麼多

        # if args.tboard:
        #     show_wc_tnsboard(epoch + 1, writer, images_val, labels_val, pred, 8, 'Val Inputs', 'Val WCs', 'Val Pred. WCs')
        #     writer.add_scalar('WC: L1 Loss/val', val_loss, epoch + 1)
        #     writer.add_scalar('WC: Grad Loss/val', val_gloss, epoch + 1)

        # val_loss = val_loss / len(valloader)
        # val_mse = val_mse / len(valloader)
        # val_gloss = val_gloss / len(valloader)
        # print("val loss at epoch {}:: {}".format(epoch + 1, val_loss))
        # print("val MSE: {}".format(val_mse))

        # val_losses = [val_loss, val_mse, val_gloss]
        # write_log_file(experiment_name, val_losses, epoch + 1, lrate, 'Val')

        # reduce learning rate
        # sched.step(val_mse)

        # if val_mse < best_val_mse:
        #     best_val_mse = val_mse
        #     state = {'epoch': epoch + 1,
        #              'model_state': model.state_dict(),
        #              'optimizer_state': optimizer.state_dict(), }
        #     torch.save(state, args.logdir + "{}_{}_{}_{}_{}_best_model.pkl".format(
        #         args.arch, epoch + 1, val_mse, train_mse, experiment_name))

        # if (epoch + 1) % 10 == 0:
        #     state = {'epoch': epoch + 1,
        #              'model_state': model.state_dict(),
        #              'optimizer_state': optimizer.state_dict(), }
        #     torch.save(state, args.logdir + "{}_{}_{}_{}_{}_model.pkl".format(
        #         args.arch, epoch + 1, val_mse, train_mse, experiment_name))

class kong_args():
    def __init__(self, arch="dnetccnl", data_path="", img_rows=256, img_cols=256, n_epoch= 100, batch_size=1, l_rate=1e-5, resume=None, logdir='./checkpoints-wc/', tboard=False,):
        self.arch       = arch
        self.data_path  = data_path
        self.img_rows   = img_rows
        self.img_cols   = img_cols
        self.n_epoch    = n_epoch
        self.batch_size = batch_size
        self.l_rate     = l_rate
        self.resume     = resume
        self.logdir     = logdir
        self.tboard     = tboard


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Hyperparams')
    # parser.add_argument('--arch', nargs='?', type=str, default='dnetccnl', help='Architecture to use [\'dnetccnl, unetnc\']')
    # parser.add_argument('--data_path', nargs='?', type=str, default='', help='Data path to load data')
    # parser.add_argument('--img_rows', nargs='?', type=int, default=256, help='Height of the input image')
    # parser.add_argument('--img_cols', nargs='?', type=int, default=256, help='Width of the input image')
    # parser.add_argument('--n_epoch', nargs='?', type=int, default=100, help='# of the epochs')
    # parser.add_argument('--batch_size', nargs='?', type=int, default=1, help='Batch Size')
    # parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5 help='Learning Rate')
    # parser.add_argument('--resume', nargs='?', type=str, default=None, help='Path to previous saved model to restart from')
    # parser.add_argument('--logdir', nargs='?', type=str, default='./checkpoints-wc/', help='Path to store the loss logs')
    # parser.add_argument('--tboard', dest='tboard', action='store_true', help='Enable visualization(s) on tensorboard | False by default')
    # parser.set_defaults(tboard=False)
    # args = parser.parse_args()

    # imgs_dir  = "G:/0 data_dir/datasets/type8_blender_os_book/blender_os_hw768/see_crop_tightly"
    # dst_dir   = "H:/0_School-108-2/paper11/DewarpNet/eval/004_DewarpNet_eval_kong_sees_crop"
    # data_path = "M:/swat3D"
    # data_path = "J:/swat3D"  ### 2022/03/30
    data_path = "J:/swat3D"  ### 2022/05/10
    default_args = kong_args(arch       = "dnetccnl",
                             data_path  = "",
                             img_rows   = 256,
                             img_cols   = 256,
                             n_epoch    = 100,
                             batch_size = 50,  ### 但是 https://github.com/cvlab-stonybrook/DewarpNet 上 寫的 training指令用的是 batch_size=50！像這樣：python trainwc.py --arch unetnc --data_path ./data/DewarpNet/doc3d/ --batch_size 50 --tboard
                             l_rate     = 1e-5,
                             resume     = None,
                             logdir     = './checkpoints-wc/',
                             tboard     = False,
                             )
    args = kong_args(arch       = "unetnc",
                     data_path  = data_path,  ### Doc3D 的位置
                     img_rows   = 256,        ### Resize 後的 height 大小
                     img_cols   = 256,        ### Resize 後的 width  大小
                     n_epoch    = 100,
                     batch_size = 50,          ### DewarpNet 附贈的程式碼 是用50， 我這邊可能為了好測試所以用1
                     l_rate     = 1e-5,
                     resume     = None,
                     logdir     = './checkpoints-wc/',
                     tboard     = True,
                    )
    train(args)

### 原始的程式碼附的
# CUDA_VISIBLE_DEVICES=1 python trainwc.py --arch unetnc --data_path ./data/DewarpNet/doc3d/ --batch_size 50 --tboard
### 我自己改的
# CUDA_VISIBLE_DEVICES=1 python trainwc.py --arch unetnc --data_path G:/swat3D/ --batch_size 50 --tboard
