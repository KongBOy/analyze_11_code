'''
Acknowledgement: https://github.com/Po-Hsun-Su/pytorch-ssim
'''
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size //
                       2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channels=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channels
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0) / 255.0  ### kong加的，從網路上copy過來的 應該是 轉 NCHW 和 弄到 0~1
    img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0) / 255.0  ### kong加的，從網路上copy過來的 應該是 轉 NCHW 和 弄到 0~1

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


if(__name__ == "__main__"):
    import cv2
    import numpy as np
    rec_dir = r"F:\kong_model2\Prepare_dataset\DewarpNet_Crop_pad20_60\DewarpNet_run"
    sca_dir = r"F:\kong_model2\Prepare_dataset\scan"

    ssim_scores = []
    for go_i in range(1, 66):
        print("current id:", go_i)
        rec_img1 = cv2.imread(f"{rec_dir}/%02i_1.png" % go_i)
        rec_img2 = cv2.imread(f"{rec_dir}/%02i_2.png" % go_i)
        sca_img = cv2.imread(f"{sca_dir}/%i.png" % go_i)
        sca_h, sca_w = sca_img.shape[:2]
        rec_img1 = cv2.resize(rec_img1, (sca_w, sca_h))
        rec_img2 = cv2.resize(rec_img2, (sca_w, sca_h))


        ssim_scores.append(ssim(sca_img, rec_img1).numpy())
        ssim_scores.append(ssim(sca_img, rec_img2).numpy())

        print("    ", ssim_scores[-2])
        print("    ", ssim_scores[-1])

    ssim_scores = np.array(ssim_scores)
    print("mean:", np.mean(ssim_scores))


'''
current id: 1
     0.52618325
     0.54817706
current id: 2
     0.574024
     0.5954935
current id: 3
     0.7024678
     0.6654296
current id: 4
     0.7052789
     0.6683451
current id: 5
     0.6719865
     0.66025305
current id: 6
     0.4408505
     0.47102186
current id: 7
     0.64559406
     0.7038535
current id: 8
     0.6206156
     0.56838495
current id: 9
     0.61441433
     0.5998039
current id: 10
     0.6433401
     0.7022506
current id: 11
     0.19749781
     0.22941336
current id: 12
     0.30405423
     0.3139942
current id: 13
     0.45309234
     0.441709
current id: 14
     0.54764074
     0.5329595
current id: 15
     0.45213142
     0.41378716
current id: 16
     0.3931491
     0.3942773
current id: 17
     0.37908533
     0.36451536
current id: 18
     0.37210464
     0.3515191
current id: 19
     0.45103812
     0.46341214
current id: 20
     0.40648237
     0.4148102
current id: 21
     0.49735203
     0.575866
current id: 22
     0.54271966
     0.5242217
current id: 23
     0.58642787
     0.57486814
current id: 24
     0.48390526
     0.4606263
current id: 25
     0.45246595
     0.47844765
current id: 26
     0.4506482
     0.47716084
current id: 27
     0.6557289
     0.6165931
current id: 28
     0.5681536
     0.5509776
current id: 29
     0.49098268
     0.46768528
current id: 30
     0.5530634
     0.551764
current id: 31
     0.43020743
     0.4729772
current id: 32
     0.63973427
     0.61992854
current id: 33
     0.608737
     0.5411897
current id: 34
     0.4326783
     0.42345124
current id: 35
     0.6644708
     0.6648168
current id: 36
     0.43200612
     0.38484284
current id: 37
     0.44319516
     0.44184458
current id: 38
     0.5186901
     0.49585387
current id: 39
     0.49233064
     0.48452973
current id: 40
     0.48125517
     0.5133312
current id: 41
     0.5312929
     0.5295874
current id: 42
     0.56673515
     0.5800489
current id: 43
     0.5089118
     0.47051784
current id: 44
     0.49217317
     0.48674324
current id: 45
     0.5027917
     0.49405473
current id: 46
     0.6813077
     0.66319317
current id: 47
     0.63455594
     0.5640555
current id: 48
     0.47962645
     0.51174754
current id: 49
     0.4509596
     0.47734696
current id: 50
     0.8064368
     0.79135615
current id: 51
     0.7267247
     0.7426377
current id: 52
     0.7736054
     0.7995134
current id: 53
     0.8123691
     0.83569705
current id: 54
     0.62724334
     0.6232341
current id: 55
     0.5277941
     0.55295604
current id: 56
     0.71651244
     0.7285393
current id: 57
     0.43885526
     0.4420591
current id: 58
     0.48330775
     0.44677386
current id: 59
     0.42646512
     0.4169563
current id: 60
     0.42076582
     0.42942902
current id: 61
     0.5663011
     0.57272816
current id: 62
     0.4954661
     0.5517048
current id: 63
     0.57387865
     0.5731658
current id: 64
     0.53481436
     0.5350718
current id: 65
     0.6659856
     0.6547977
mean: 0.53745353
'''