import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random
import pdb

# root='/media/hilab/sagniksSSD/Sagnik/DewarpNet/swat3d/'
# filenames=['7/2_427_8-cp_Page_1362-4rw0001','7/2_87_5-ec_Page_040-AZI0001','7/1_811_3-ny_Page_554-sof0001','7/2_168_4-ny_Page_888-qoK0001',
#            '7/1_50_7-ns_Page_527-fyQ0001','7/827_6-ny_Page_040-x410001','7/762_7-ns_Page_579-UzD0001','7/2_456_6-pp_Page_278-tUY0001',
#            '7/1_996_5-ns_Page_402-icm0001','7/2_85_5-ns_Page_523-rgf0001']


def tight_crop(im, fm):
    '''
    因為 pytorch 的 Dataset 好像沒辦法用 plt.imshow() 和 breakpoint， 所以只好 用 plt.savefig 和 VSCode的 中斷點來 分析 囉～
    '''
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    # different tight crop
    msk = ((fm[:, :, 0] != 0) & (fm[:, :, 1] != 0) & (fm[:, :, 2] != 0)).astype(np.uint8)
    ax[0].set_title("ord_img")
    ax[0].imshow(im)
    ax[1].set_title("mask")
    ax[1].imshow(msk)

    # print("msk.shape", msk.shape)  ### (448, 448)
    # print(im.shape)  ### (448, 448, 3)  0 ~ 1
    # print(fm.shape)  ### (448, 448, 3) -1.174 ~ 1.1318
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    im = im[miny : maxy + 1, minx : maxx + 1, :]
    fm = fm[miny : maxy + 1, minx : maxx + 1, :]
    ax[2].set_title("ord_img_croped")
    ax[2].imshow(im)

    # px = int((maxx - minx) * 0.07)
    # py = int((maxy - miny) * 0.07)

    # im = np.pad(im, ((py, py + 1), (px, px + 1), (0, 0)), 'constant')
    # fm = np.pad(fm, ((py, py + 1), (px, px + 1), (0, 0)), 'constant')
    # # crop
    # cx1 = int(random.randint(0, 3) / 7.0 * px)
    # cx2 = int(random.randint(0, 3) / 7.0 * px + 1)
    # cy1 = int(random.randint(0, 3) / 7.0 * py)
    # cy2 = int(random.randint(0, 3) / 7.0 * py + 1)

    s = 20
    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    fm = np.pad(fm, ((s, s), (s, s), (0, 0)), 'constant')
    cx1 = random.randint(0, s - 5)
    cx2 = random.randint(0, s - 5) + 1
    cy1 = random.randint(0, s - 5)
    cy2 = random.randint(0, s - 5) + 1

    ax[3].set_title("ord_img_crop_&_pad_&_jit")
    ax[3].imshow(im)
    plt.savefig("analyze_by_kong/tight_crop")

    print("cx1", cx1)
    print("cx2", cx2)
    print("cy1", cy1)
    print("cy2", cy2)
    print("im.shape", im.shape)

    im = im[cy1 : -cy2, cx1 : -cx2, :]
    fm = fm[cy1 : -cy2, cx1 : -cx2, :]
    print("im.shape", im.shape)
    return im, fm


def tight_crop_d(im, dm):
    # different tight crop
    msk = (dm != 0).astype(np.uint8)
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    im = im[miny : maxy + 1, minx : maxx + 1, :]
    dm = dm[miny : maxy + 1, minx : maxx + 1]

    # px = int((maxx - minx) * 0.07)
    # py = int((maxy - miny) * 0.07)

    # im = np.pad(im, ((py, py + 1), (px, px + 1), (0, 0)), 'constant')
    # fm = np.pad(fm, ((py, py + 1), (px, px + 1), (0, 0)), 'constant')
    # # crop
    # cx1 = int(random.randint(0, 3) / 7.0 * px)
    # cx2 = int(random.randint(0, 3) / 7.0 * px + 1)
    # cy1 = int(random.randint(0, 3) / 7.0 * py)
    # cy2 = int(random.randint(0, 3) / 7.0 * py + 1)

    s = 20
    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    dm = np.pad(dm, ((s, s), (s, s)), 'constant')
    cx1 = random.randint(0, s - 5)
    cx2 = random.randint(0, s - 5) + 1
    cy1 = random.randint(0, s - 5)
    cy2 = random.randint(0, s - 5) + 1

    im = im[cy1 : -cy2, cx1 : -cx2, :]
    dm = dm[cy1 : -cy2, cx1 : -cx2]
    return im, dm


def color_jitter(im, brightness=0, contrast=0, saturation=0, hue=0):
    f = random.uniform(1 - contrast, 1 + contrast)
    im = np.clip(im * f, 0., 1.)
    f = random.uniform(-brightness, brightness)
    im = np.clip(im + f, 0., 1.).astype(np.float32)

    # hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    # f = random.uniform(-hue, hue)
    # hsv[0] = np.clip(hsv[0] + f * 360, 0., 360.)
    # f = random.uniform(-saturation, saturation)
    # hsv[2] = np.clip(hsv[2] + f, 0., 1.)
    # im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # im = np.clip(im, 0., 1.)
    return im


def change_intensity(img):
    chance = random.uniform(0, 1)
    # print(chance)
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if chance > 0.3:
        inc = random.randint(15, 50)
        # print(inc)
        # increase
        v = nimg[:, :, 2]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 2] = v

    nimg = cv2.cvtColor(nimg, cv2.COLOR_HSV2BGR)
    # f,axarr=plt.subplots(1,2)
    # axarr[0].imshow(img)
    # axarr[1].imshow(nimg)
    # plt.show()
    return nimg


def change_hue_sat(img):
    chance = random.uniform(0, 1)
    # print(chance)
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if chance > 0.3:
        inc = random.randint(5, 15)
        # print(inc)
        # increase
        v = nimg[:, :, 0]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 0] = v

    if chance > 0.3:
        inc = random.randint(5, 15)
        # print(inc)
        # increase
        v = nimg[:, :, 1]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 1] = v

    nimg = cv2.cvtColor(nimg, cv2.COLOR_HSV2BGR)
    # f,axarr=plt.subplots(1,2)
    # axarr[0].imshow(img)
    # axarr[1].imshow(nimg)
    # plt.show()
    return nimg


def data_aug(im, fm, bg):
    '''
    因為 pytorch 的 Dataset 好像沒辦法用 plt.imshow() 和 breakpoint， 所以只好 用 plt.savefig 和 VSCode的 中斷點來 分析 囉～
    '''
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    # print("first im to 0~1")
    im = im / 255.0
    # print("first im.max()", im.max())
    # print("first im.min()", im.min())
    bg = bg / 255.0
    if fm.shape[-1] == 3:
        im, fm = tight_crop(im, fm)
        print("tight_crop finish")
    else:
        im, fm = tight_crop_d(im, fm)
        print("tight_crop_d finish")
    # change background img
    # msk = fm[:, :, 0] > 0
    if fm.shape[-1] == 3:
        msk = ((fm[:, :, 0] != 0) & (fm[:, :, 1] != 0) & (fm[:, :, 2] != 0)).astype(np.uint8)  ### shape 比如 (294, 214)
    else:
        msk = (fm != 0).astype(np.uint8)
    print("msk.shape", msk.shape)
    msk = np.expand_dims(msk, axis=2)  ### shape 比如 (294, 214, 1)
    print("msk.shape", msk.shape)

    # replace bg
    [fh, fw, _] = im.shape
    chance = random.random()
    if chance > 0.3:  ### texture背景
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))
        bg = bg[: fh, : fw, :]
    elif chance < 0.3 and chance > 0.2:  ### 單色背景
        c = np.array([random.random(), random.random(), random.random()])
        bg = np.ones((fh, fw, 3)) * c
    else:  ### 全黑背景， 就是 無背景拉
        bg = np.zeros((fh, fw, 3))
        msk = np.ones((fh, fw, 3))
    im = bg * (1 - msk) + im * msk
    # jitter color
    im = color_jitter(im, 0.2, 0.2, 0.6, 0.6)
    # im = change_hue_sat(im)
    # im = change_intensity(im)

    # plt.imshow(im)
    # plt.show()
    # plt.imshow(fm)
    # plt.show()
    return im, fm


# def main():
#     tex_id=random.randint(1,5640)
#     with open(os.path.join(root[:-7],'augtexnames.txt'),'r') as f:
#         for i in range(tex_id):
#             txpth=f.readline().strip()

#     for im_name in filenames:

#         im_path = os.path.join(root,'img',im_name+'.png')
#         img=cv2.imread(im_path).astype(np.uint8)

#         lbl_path = os.path.join(root, 'wc',im_name+'.exr')
#         lbl = cv2.imread(lbl_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

#         tex=cv2.imread(os.path.join(root[:-7],txpth)).astype(np.uint8)
#         bg=cv2.resize(tex,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LANCZOS4)

#         img,lbl=data_aug(img,lbl,bg)

# if __name__ == '__main__':
#     main()
