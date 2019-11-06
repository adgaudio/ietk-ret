from sklearn.decomposition import FastICA, PCA, NMF
import cv2
import numpy as np
import util


# improve calculation speed by down scaling image before finding bases especially for NMF
SCALE_RATE = 0.5


def down_scale(img):
    width = int(img.shape[1] * SCALE_RATE)
    height = int(img.shape[0] * SCALE_RATE)
    resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    return resized


def pca(img):
    assert (len(img.shape) == 3)
    img_fit = down_scale(img)
    img_fit = img_fit.reshape(-1, 3)

    # run pca
    model = PCA(n_components=3)
    model.fit_transform(img_fit)

    # get result
    height, width, _ = img.shape
    img = img.reshape(-1, 3)
    result = model.transform(img)
    result = np.reshape(result, (height, width, 3))

    # abs the result thus the min operation will find the distance to each base
    result = abs(result)

    return result


def ica(img):
    assert (len(img.shape) == 3)
    img_fit = down_scale(img)
    img_fit = img_fit.reshape(-1, 3)

    # run ica
    model = FastICA(n_components=3)
    model.fit_transform(img_fit)

    # get result
    height, width, _ = img.shape
    img = img.reshape(-1, 3)
    result = model.transform(img)
    result = np.reshape(result, (height, width, 3))

    # abs the result thus the min operation will find the distance to each base
    result = abs(result)

    return result


def ica2(img, background):
    tmp = FastICA(3).fit_transform(img.reshape(-1, 3)).reshape(img.shape)
    rv = util.norm01(tmp, background)
    return rv


def nmf(img):
    assert (len(img.shape) == 3)
    img_fit = down_scale(img)
    img_fit = img_fit.reshape(-1, 3)

    # run NMF
    model = NMF(n_components=3, init='random', random_state=0)
    model.fit_transform(img_fit)

    # get result
    height, width, _ = img.shape
    img = img.reshape(-1, 3)
    result = model.transform(img)
    result = np.reshape(result, (height, width, 3))

    return result


def cmy_from_rgb(rgb_img):
    """demo example of how to pose change of basis as a rotation"""
    coeffs = np.array([[0, .5, .5],  # cyan
                       [.5, 0, .5],  # magenta
                       [.5, .5, 0]]  # yellow
                      ).T
    cmy_img = (rgb_img.reshape(-1, 3) @ coeffs).reshape(rgb_img.shape)

    # check:
    #  r = rgb_img[:, :, 0]
    #  g = rgb_img[:, :, 1]
    #  b = rgb_img[:, :, 2]
    #  test_cmy = np.dstack([(b+g)/2, (r+b)/2, (r+g)/2])
    #  assert ((test_cmy - cmy_img) == 0).all()

    # check 2: total energy is preserved
    np.testing.assert_allclose(rgb_img.sum(), cmy_img.sum())
    return cmy_img


def rgb_from_cmy(cmy_img):
    """demo example, part two"""
    coeffs = np.linalg.inv(
        np.array([[0, .5, .5],  # cyan
                  [.5, 0, .5],  # magenta
                  [.5, .5, 0]]  # yellow
                 ).T)
    rgb_img = cmy_img @ coeffs
    np.testing.assert_allclose(rgb_img.sum(), cmy_img.sum())
    return rgb_img
