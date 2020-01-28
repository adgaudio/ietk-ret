"""
Retinex image enhancement method.  Retinex is equivalent to inverted dehazing.
Code downloaded from https://github.com/falrom/MSRCR_Python/blob/master/MSRCR.py
on 2019-11-01 and slightly modified.
"""
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import argparse
import cv2
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input image path')
    parser.add_argument('-o', '--output', required=True, help='output image path')
    parser.add_argument('-s', default=300, type=float, help='the scale (reference value)')
    parser.add_argument('-n', default=3, type=int, help='the number of scale')
    parser.add_argument('-d', default=2, type=float, help='the dynamic, the smaller the value, the higher the contrast')
    parser.add_argument('--no_cr', action='store_true', help='do NOT do cr')


def retinex_scales_distribution(max_scale, nscales):
    scales = []
    scale_step = max_scale / nscales
    for s in range(nscales):
        scales.append(scale_step * s + 2.0)
    return scales


def CR(im_ori, im_log, alpha=128., gain=1., offset=0.):
    im_cr = im_log * gain * (
            np.ma.log(alpha * (im_ori + 1.0)) - np.ma.log(np.ma.sum(im_ori, axis=2) + 3.0)[:, :, np.newaxis]) + offset
    return im_cr


def blur(img_1channel, sigma):
    #  rv = ndi.gaussian_filter(img_1channel, sigma)
    # faster blur
    img = (img_1channel/255).astype('float32')
    rv = cv2.ximgproc.guidedFilter(img, img, int(sigma), 1e-2) * 255
    return rv


def MSRCR(rgb_img, max_scale, nscales, dynamic=2.0, do_CR=True):
    assert rgb_img.max() > 1
    im_ori = rgb_img
    scales = retinex_scales_distribution(max_scale, nscales)

    im_blur = np.zeros([len(scales), im_ori.shape[0], im_ori.shape[1], im_ori.shape[2]])
    im_mlog = np.zeros([len(scales), im_ori.shape[0], im_ori.shape[1], im_ori.shape[2]])

    for channel in range(3):
        for s, scale in enumerate(scales):
            # If sigma==0, it will be automatically calculated based on scale
            im_blur[s, :, :, channel] = blur(im_ori[:, :, channel], scale)
            im_mlog[s, :, :, channel] = np.ma.log(im_ori[:, :, channel] + 1.) - np.log(im_blur[s, :, :, channel] + 1.)

    im_retinex = np.mean(im_mlog, 0)
    im_retinex = np.ma.masked_array(im_retinex, im_ori.mask)
    if do_CR:
        im_retinex = CR(im_ori, im_retinex)

    im_rtx_mean = np.ma.mean(im_retinex)
    im_rtx_std = np.ma.std(im_retinex)
    im_rtx_min = im_rtx_mean - dynamic * im_rtx_std
    im_rtx_max = im_rtx_mean + dynamic * im_rtx_std

    im_rtx_range = im_rtx_max - im_rtx_min

    im_out = np.uint8(np.ma.clip((im_retinex - im_rtx_min) / im_rtx_range * 255.0, 0, 255))

    return im_out


if __name__ == '__main__':
    ####################################################################################
    # plt.close('all')
    # image_path = r'test_images/18.jpg'
    # out_msrcr = MSRCR(image_path, max_scale=300, nscales=3, dynamic=2, do_CR=True)
    # plt.figure(); plt.title('MSRCR'); plt.imshow(out_msrcr)
    # out_msr = MSRCR(image_path, max_scale=300, nscales=3, dynamic=2, do_CR=False)
    # plt.figure(); plt.title('MSR'); plt.imshow(out_msr)
    # plt.show()
    ####################################################################################
    args = parser.parse_args()
    import util
    im_in = plt.imread(args.input).copy().astype('float32')
    im_in = np.ma.masked_array(im_in, (util.get_background(im_in/255)))
    im_out = MSRCR(im_in, args.s, args.n, args.d, not args.no_cr)
    cv2.imwrite(args.output, im_out[:, :, (2, 1, 0)])
