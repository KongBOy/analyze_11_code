# test end to end benchmark data test
### 我自己分析畫出的流程圖 infer.py
### https://drive.google.com/file/d/1muSsMigSQGIhpZG7vEsq_d-JK7vmnv8X/view?usp=sharing
import sys
sys.path.append("C:/Users/TKU/Desktop/kong_model2/kong_util")
from matplot_fig_ax_util import check_fig_ax_init, img_scatter_visual
from flow_bm_util import dis_bm_rec_visual, bm_arrow_visual
from build_dataset_combine import Check_dir_exist_and_build

import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.autograd import Variable
import matplotlib.pyplot as plt


from models import get_model
from utils import convert_state_dict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

debug_dict = {}


def unwarp(img, bm):
    w, h = img.shape[0], img.shape[1]  ### ord_h=768, ord_w=768
    bm = bm.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()[0, :, :, :]
    bm0 = cv2.blur(bm[:, :, 0], (3, 3))
    bm1 = cv2.blur(bm[:, :, 1], (3, 3))
    bm0 = cv2.resize(bm0, (h, w))
    bm1 = cv2.resize(bm1, (h, w))
    bm = np.stack([bm0, bm1], axis=-1)
    bm = np.expand_dims(bm, 0)
    bm = torch.from_numpy(bm).double()
    debug_dict["step3_3 bm"] = bm.numpy()[0]

    img = img.astype(float) / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).double()
    debug_dict["step3_4 img"] = img.numpy().transpose(0, 2, 3, 1)[0]  # NCHW  -> NHWC -> HWC

    res = F.grid_sample(input=img, grid=bm)
    res = res[0].numpy().transpose((1, 2, 0))
    debug_dict["step3_5 res"] = res

    return res


def test(args, img_path, fname):
    ###############################################################
    ### load wc model
    wc_n_classes = 3
    wc_img_size = (256, 256)
    wc_model_file_name = os.path.split(args.wc_model_path)[1]
    wc_model_name = wc_model_file_name[:wc_model_file_name.find('_')]
    wc_model = get_model(wc_model_name, wc_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        wc_state = convert_state_dict(torch.load(
            args.wc_model_path, map_location='cpu')['model_state'])
    else:
        wc_state = convert_state_dict(
            torch.load(args.wc_model_path)['model_state'])
    wc_model.load_state_dict(wc_state)
    wc_model.eval()


    ###############################################################
    ### load bm model
    bm_n_classes = 2
    bm_img_size = (128, 128)
    bm_model_file_name = os.path.split(args.bm_model_path)[1]
    bm_model_name = bm_model_file_name[:bm_model_file_name.find('_')]
    bm_model = get_model(bm_model_name, bm_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        bm_state = convert_state_dict(torch.load(
            args.bm_model_path, map_location='cpu')['model_state'])
    else:
        bm_state = convert_state_dict(
            torch.load(args.bm_model_path)['model_state'])
    bm_model.load_state_dict(bm_state)
    bm_model.eval()


    ###############################################################
    ### Setup image
    print("Read Input Image from : {}".format(img_path))
    imgorg = cv2.imread(img_path)
    imgorg = cv2.cvtColor(imgorg, cv2.COLOR_BGR2RGB)
    img = cv2.resize(imgorg, wc_img_size)
    img = img[:, :, ::-1]
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)  # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    debug_dict["step1 imgorg"] = imgorg
    debug_dict["step1 img"]    = img.numpy().transpose(0, 2, 3, 1)[0]  # NCHW  -> NHWC -> HWC

    ###############################################################
    ### set CPU/GPU
    if torch.cuda.is_available():
        wc_model.cuda()
        bm_model.cuda()
        images = Variable(img.cuda())
    else:
        images = Variable(img)

    ###############################################################
    ### Predict
    htan = nn.Hardtanh(0, 1.0)
    with torch.no_grad():
        wc_outputs = wc_model(images)
        pred_wc = htan(wc_outputs)
        bm_input = F.interpolate(pred_wc, bm_img_size)
        outputs_bm = bm_model(bm_input)

    ### call unwarp
    uwpred = unwarp(imgorg, outputs_bm)


    debug_dict["step2_1 wc_outputs"] = wc_outputs.cpu().numpy().transpose(0, 2, 3, 1)[0]  # NCHW  -> NHWC -> HWC
    debug_dict["step2_2 pred_wc"]    = pred_wc.cpu().numpy().transpose(0, 2, 3, 1)[0]  # NCHW  -> NHWC -> HWC
    debug_dict["step2_3 bm_input"]   = bm_input.cpu().numpy().transpose(0, 2, 3, 1)[0]  # NCHW  -> NHWC -> HWC
    debug_dict["step2_4 outputs_bm"] = outputs_bm.cpu().numpy().transpose(0, 2, 3, 1)[0]  # NCHW  -> NHWC -> HWC

    debug_dict["step3_1 imgorg"]     = imgorg
    debug_dict["step3_2 outputs_bm"] = outputs_bm.cpu().numpy().transpose(0, 2, 3, 1)[0]  # NCHW  -> NHWC -> HWC
    debug_dict["step3_end uwpred"]   = uwpred
    ###############################################################
    ### matplot visual
    if args.show:
        f1, axarr1 = plt.subplots(1, 2)
        axarr1[0].imshow(imgorg)
        axarr1[1].imshow(uwpred)
        plt.show()

    if args.kong_show:
        fig, ax, ax_c = dis_bm_rec_visual(debug_dict["step1 imgorg"], debug_dict["step2_4 outputs_bm"], uwpred, img_smaller=0.4,
                                        x_min= -1.00, x_max=+1.00, y_min=-1.00, y_max=+1.00,
                                        jump_r=3, jump_c=3,
                                        dis_alpha=0.6, dis_dot_s=1,
                                        bm_arrow_alpha=0.5, bm_arrow_cmap="hsv",
                                        bm_after_dot_alpha=0.35, bm_after_dot_cmap="hsv",
                                        tight_layout=True)
        name = fname.split(".")[0]
        plt.savefig(args.out_path + "/dis_bm_rec/" + name + "_dis_bm_rec")
        plt.close()

    # Save the output
    outp = os.path.join(args.out_path, fname)
    cv2.imwrite(outp, uwpred[:, :, ::-1] * 255)


class kong_args():
    def __init__(self, wc_model_path, bm_model_path, img_path, out_path, show, kong_show):
        self.wc_model_path = wc_model_path
        self.bm_model_path = bm_model_path
        self.img_path = img_path
        self.out_path = out_path
        self.show = show
        self.kong_show = kong_show


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--wc_model_path', nargs='?', type=str, default='', help='Path to the saved wc model')
    parser.add_argument('--bm_model_path', nargs='?', type=str, default='', help='Path to the saved bm model')
    parser.add_argument('--img_path', nargs='?', type=str, default='./eval/inp/', help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default='./eval/uw/', help='Path of the output unwarped image')
    parser.add_argument('--show', dest='show', action='store_true', help='Show the input image and output unwarped')
    parser.set_defaults(show=False)
    args = parser.parse_args()

    # imgs_dir = "H:/0_School-108-1/paper09/Benchmark Dataset/crop"
    # dst_dir  = "H:/0_School-108-2/paper11/DewarpNet/eval/002-DewarpNet_eval_DocUNet_Benchmark_crop"

    # imgs_dir = "G:/0 data_dir/datasets/type8_blender_os_book/blender_os_hw768/see/dis_imgs"
    # dst_dir  = "H:/0_School-108-2/paper11/DewarpNet/eval/003_DewarpNet_eval_kong_sees"

    imgs_dir = "G:/0 data_dir/datasets/type8_blender_os_book/blender_os_hw768/see_crop_tightly"
    dst_dir  = "H:/0_School-108-2/paper11/DewarpNet/eval/004_DewarpNet_eval_kong_sees_crop"

    Check_dir_exist_and_build(dst_dir)
    args = kong_args(wc_model_path = "H:/0_School-108-2/paper11/DewarpNet/dewarpnet_public_models/unetnc_doc3d.pkl",
                     bm_model_path = "H:/0_School-108-2/paper11/DewarpNet/dewarpnet_public_models/dnetccnl_doc3d.pkl",
                     img_path      = imgs_dir,
                     out_path      = dst_dir,
                     show          = False,
                     kong_show     = True,
                     )
    if(args.kong_show):
        Check_dir_exist_and_build(dst_dir + "/dis_bm_rec")
    for go_f, fname in enumerate(os.listdir(args.img_path)[:]):
        print("%03i" % go_f, end=" ")
        if '.jpg' in fname or '.JPG' in fname or '.png' in fname:
            img_path = os.path.join(args.img_path, fname)
            test(args, img_path, fname)


# python infer.py --wc_model_path ./eval/models/unetnc_doc3d.pkl --bm_model_path ./eval/models/dnetccnl_doc3d.pkl --show
