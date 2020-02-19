from scipy import stats


def ks_test_max_per_channel(img, mask, focus_region):
    """Compute a 2-sample Kolmogorov-Smirnov statistic on each channel of image
    returning the max value across channels.  Ignore the background regions

    img - array of shape (x, y, z) with z being the channels.
    mask - bool array separating healthy from diseased pixels
    focus_region - bool array containing the area to not ignore.

    Return float in [0, 1]
        0 if the healthy and diseased pixels have the same distribution,
        1 if they come from different distribution.

    Input:
        img - array of shape (h,w,ch)
            Input image, which may have a variable number of channels
        mask - boolean array of shape (h, w)
            Ground truth labels identifying healthy vs diseased pixels.
    """
    bg = ~focus_region
    assert mask.dtype == 'bool'
    assert mask.shape == img.shape[:2]
    maxstat = 0
    for ch in range(img.shape[2]):
        # get diseased (a) and healthy (b) pixels
        a = img[:, :, ch][mask & (~bg[:, :, ch])]
        b = img[:, :, ch][(~mask) & (~bg[:, :, ch])]
        # compute the stat
        statistic = stats.ks_2samp(a.ravel(), b.ravel())[0]
        maxstat = max(statistic, maxstat)
    return maxstat


single_image_eval_methods = {
    'KS Test, max of the channels': ks_test_max_per_channel  # test if healthy separable from diseased.
    # need test if lesions (MA,HE,etc) have the same intensities across images.  how about:  var(sum_imgs(p(rgb|lesion=MA)))  where p(rgb|lesion) is a 256*3 vector for each img.
}
