import cv2
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt


def sobel(window_size):
    assert(window_size % 2 != 0)
    ind = window_size // 2
    matx = []
    maty = []
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gx_ij = 0
            else:
                gx_ij = i / float(i * i + j * j)
            row.append(gx_ij)
        matx.append(row)
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gy_ij = 0
            else:
                gy_ij = j / float(i * i + j * j)
            row.append(gy_ij)
        maty.append(row)

    # matx=[[-3, 0,+3],
    #       [-10, 0 ,+10],
    #       [-3, 0,+3]]
    # maty=[[-3, -10,-3],
    #       [0, 0 ,0],
    #       [3, 10,3]]
    if window_size == 3:
        mult = 2
    elif window_size == 5:
        mult = 20
    elif window_size == 7:
        mult = 780

    matx = np.array(matx) * mult
    maty = np.array(maty) * mult

    return torch.Tensor(matx), torch.Tensor(maty)


def create_window(window_size, channel):
    windowx, windowy = sobel(window_size)
    windowx, windowy = windowx.unsqueeze(0).unsqueeze(0), windowy.unsqueeze(0).unsqueeze(0)
    windowx = torch.Tensor(windowx.expand(channel, 1, window_size, window_size))  ### shape 1,1,5,5
    windowy = torch.Tensor(windowy.expand(channel, 1, window_size, window_size))  ### shape 1,1,5,5
    # print windowx
    # print windowy

    return windowx, windowy


def gradient(img, windowx, windowy, window_size, padding, channel):
    if channel > 1:		# do convolutions on each channel separately and then concatenate
        gradx = torch.ones(img.shape)
        grady = torch.ones(img.shape)
        if img.is_cuda:
            gradx = gradx.cuda(img.get_device())
            grady = grady.cuda(img.get_device())
        # print(gradx[:,0,:,:].shape)
        for i in range(channel):
            gradx[:, i, :, :] = F.conv2d(img[:, i, :, :].unsqueeze(1), windowx, padding=padding, groups=1).squeeze(1)  # fix the padding according to the kernel size
            grady[:, i, :, :] = F.conv2d(img[:, i, :, :].unsqueeze(1), windowy, padding=padding, groups=1).squeeze(1)

    else:
        gradx = F.conv2d(img, windowx, padding=padding, groups=1)
        grady = F.conv2d(img, windowy, padding=padding, groups=1)

    return gradx, grady


class Gradloss(torch.nn.Module):
    def __init__(self, window_size=3, padding=1):
        super(Gradloss, self).__init__()
        self.window_size = window_size
        self.padding = padding
        self.channel = 1			# out channel
        self.windowx, self.windowy = create_window(window_size, self.channel)

    def forward(self, pred, label):
        (batch_size, channel, _, _) = pred.size()
        if pred.is_cuda:
            self.windowx = self.windowx.cuda(pred.get_device())
            self.windowx = self.windowx.type_as(pred)
            self.windowy = self.windowy.cuda(pred.get_device())
            self.windowy = self.windowy.type_as(pred)

        pred_gradx, pred_grad_y   = gradient(pred, self.windowx, self.windowy, self.window_size, self.padding, channel)
        label_gradx, label_grad_y = gradient(label, self.windowx, self.windowy, self.window_size, self.padding, channel)
        # label_grad=torch.sqrt((label_gradx*label_gradx) + (label_grad_y*label_grad_y))
        # w=((label_grad[:,0,:,:]>=1)&(label_grad[:,1,:,:]>=1)&(label_grad[:,2,:,:]>=1)).float()*0.7
        # msk=((label[:,0,:,:]!=0)&(label[:,1,:,:]!=0)&(label[:,2,:,:]!=1)).float()
        # w+=msk*0.2
        # w+=(1-msk)*0.1
        # w=w.expand_as(pred)

        l1_loss = nn.L1Loss()
        # l2_loss=nn.MSELoss()
        grad_loss = l1_loss(pred_gradx, label_gradx) + \
                    l1_loss(pred_grad_y, label_grad_y)
        # w_grad_loss=(label-pred)**2
        # w_grad_loss=w*w_grad_loss
        # w_grad_loss=torch.mean(w_grad_loss)
        return grad_loss, label_gradx, label_grad_y

# For testing
if __name__ == '__main__':
    window_size = 3
    padding = int((window_size - 1) / 2)
    img1_path = "1_1_8-pp_Page_465-YHc0001.exr"  ### "1_1_2-cp_Page_0654-XKI0001.exr"
    img2_path = "1_1_1-pr_Page_141-PZU0001.exr"  ### "1_1_1-tc_Page_065-YGB0001.exr"
    img1 = cv2.imread(img1_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img2 = cv2.imread(img2_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(f"img1.shape={img1.shape}")
    print(f"img2.shape={img2.shape}")

    # OpenCV sobel gradient for to check correctness
    ### cv2.sobel(img, dtype?????????, ????????????dx, ????????????dy, k_size)  https://www.itread01.com/content/1542925983.html
    sobelx1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=window_size)
    sobely1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=window_size)
    sobelx2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=window_size)
    sobely2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=window_size)
    sobely3 = cv2.Sobel(img2, cv2.CV_64F, 1, 1, ksize=window_size)

    img1 = np.array(img1, dtype=np.float).transpose(2, 0, 1)  ### CHW, (3, 488, 488)
    img2 = np.array(img2, dtype=np.float).transpose(2, 0, 1)  ### CHW, (3, 488, 488)
    img1_tv = -img1[:, -1:, :] + img1[:, 1:, :]
    img1_tv = img1_tv.transpose(1, 2, 0)
    img1 = torch.from_numpy(img1).float().unsqueeze(0)        ### NCHW, (1, 3, 488, 488)
    img2 = torch.from_numpy(img2).float().unsqueeze(0)        ### NCHW, (1, 3, 488, 488)

    gradloss = Gradloss(window_size=window_size, padding=padding)

    same_gloss, label_gradx, label_grady = gradloss(img1, img1)
    gradx1 = np.array(label_gradx[0]).transpose(1, 2, 0)
    grady1 = np.array(label_grady[0]).transpose(1, 2, 0)

    diff_gloss, label_gradx, label_grady = gradloss(img1, img2)
    gradx2 = np.array(label_gradx[0]).transpose(1, 2, 0)
    grady2 = np.array(label_grady[0]).transpose(1, 2, 0)

    show_size = 4.5
    f, axarr = plt.subplots(2, 5, figsize=(show_size * 4, show_size * 2))
    axarr[0][0].imshow(sobelx1)
    axarr[0][1].imshow(sobely1)
    axarr[0][2].imshow(sobelx2)
    axarr[0][3].imshow(sobely2)
    axarr[0][4].imshow(img1_tv)

    axarr[1][0].imshow(gradx1)
    axarr[1][1].imshow(grady1)
    axarr[1][2].imshow(gradx2)
    axarr[1][3].imshow(grady2)
    plt.tight_layout()
    plt.show()

    print(same_gloss.item())
    print(diff_gloss.item())
