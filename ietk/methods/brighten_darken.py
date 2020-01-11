import scipy.ndimage as ndi
import numpy as np
from cv2.ximgproc import guidedFilter
from matplotlib import pyplot as plt

from ietk.methods import dehaze
from ietk import util
from ietk.data import IDRiD


def sh(I, num=None, title=None):
    plt.figure(num=num) ; plt.imshow(I.squeeze())
    plt.title(title)


def solveJ(I, A, t):
    return (I-A)/t + A


def solvet(I, A):
    z = 1-ndi.minimum_filter((I/A).min(-1), (5, 5))
    return gf(I, z).reshape(*I.shape[:2], 1)


def solvetmax(I, A):
    z = 1-ndi.maximum_filter((I/A).max(-1), (5, 5))
    return gf(I, z).reshape(*I.shape[:2], 1)


def gf(guide, src, r=100, eps=1e-8):
    return guidedFilter(guide.astype('float32'), src.astype('float32'), r, eps).astype('float64')


def brighten_dark_areas_v1(I):
    """ brighten the DARK areas"""
    #  t = 1-solvetmax(I, 1)  # same as below
    t = solvet(1-I, 1)
    J = solveJ(I, 0, t)
    return J

def brighten_dark_areas_v2(I):
    """ brighten the DARK areas"""
    t2 = solvet(1-I, 1)
    J = 1-solveJ(1-I, 1, t2)
    return J

def brighten_bright_areas_v1(I):
    """ brighten the BRIGHT areas."""
    t = solvetmax(I, 1)
    return solveJ(I, 0, t)

def brighten_bright_areas_v2(I):
    """ brighten the BRIGHT areas."""
    t = 1-solvet(1-I, 1)
    return solveJ(I, 0, t)

def darken_bright_areas_redgreen_v1(I):
    """ darken the BRIGHT areas (ignoring blue channel, which is necessary for
    retinal fundus images)"""
    I2 = I[:,:,:2]  # red and green channels only
    t2 = solvet(I2, 1)
    J2 = solveJ(I2, 1, t2)
    return np.dstack([J2, I[:,:,2]])  # use original blue channel


def darken_dark_areas_redgreen_v1(I):
    """ darken the DARK areas (ignoring blue channel, which is necessary for
    retinal fundus images)"""
    I2 = I[:,:,:2]  # red and green channels only
    t = solvet(1-I2, 1)
    J = solveJ(I, 1, t)
    return J

def darken_dark_areas_redgreen_v2(I):
    """ darken the DARK areas (ignoring blue channel, which is necessary for
    retinal fundus images)"""
    # normal dehazing.
    # can't use the blue channel in retinal fundus imgs because all black.
    #  t = solvet(I[:,:,:2], 1)
    I2 = I[:,:,:2]
    #  t2 = 1-solvetmax(I2, 1)
    t2 = solvet(1-I2, 1)

    J = 1-solveJ(1-I, 0, t2)
    return J
    # or just on red and green:
    #  J2 = 1-solveJ(1-I2, 0, t2)
    #  return np.dstack([J2, I[:,:,2]])


def compose_darken(I):
    """  darken both bright and dark regions"""
    I2 = I[:,:,:2]  # red and green channels only

    t1 = solvet(I2, 1)
    A1 = 1
    t2 = solvet(1-I2, 1)
    A2 = 1
    return solveJ(solveJ(I, A1, t1), A2, t2)

def compose_brighten(I):
    """ brighten bright and dark areas"""
    # compare to running a pipeline:
    # sh(brighten_bright_areas_v2(brighten_dark_areas_v2(I)), 7)
    t1 = 1-solvetmax(I, 1)
    A1 = 0
    t2 = 1-solvet(1-I, 1)
    A2 = 0
    return solveJ(solveJ(I, A1, t1), A2, t2)


def compose_bd(I):
    """  darken and then brighten both bright and dark regions"""
    I2 = I[:,:,:2]  # red and green channels only
    # bright
    t1 = solvet(I2, 1)
    A1 = 1
    t2 = solvet(1-I2, 1)
    A2 = 1

    # dark
    t3 = 1-solvetmax(I, 1)
    A3 = 0
    t4 = 1-solvet(1-I, 1)
    A4 = 0
    return solveJ(solveJ(solveJ(solveJ(I, A1, t1), A2, t2), A3, t3), A4, t4)


def compose_db(I):
    """ brighten and then darken both bright and dark regions"""

    # shows the usefulness of composition by comparing to:  compose_brighten(compose_darken(I))
    I2 = I[:,:,:2]  # red and green channels only
    # bright
    t1 = solvet(I2, 1)
    A1 = 1
    t2 = solvet(1-I2, 1)
    A2 = 1

    # dark
    t3 = 1-solvetmax(I, 1)
    A3 = 0
    t4 = 1-solvet(1-I, 1)
    A4 = 0
    return solveJ(solveJ(solveJ(solveJ(I, A3, t3), A4, t4), A2, t2), A1, t1)


def compose_bd_simpler(I):
    """ darken the bright areas and then brighten the dark areas"""
    t1 = solvet(1-I, 1)  #a
    A1 = 0
    I2 = I[:,:,:2]
    t2 = solvet(I2, 1)  #d
    A2 = 1
    return solveJ(solveJ(I, A1, t1), A2,t2)


def compose_db_simpler(I):
    """ brighten the dark areas and then darken the bright areas"""
    t1 = solvet(1-I, 1)  #a
    A1 = 0
    I2 = I[:,:,:2]
    t2 = solvet(I2, 1)  #d
    A2 = 1
    return solveJ(solveJ(I, A2, t2), A1,t1)


if __name__ == "__main__":
    from ietk import methods
    from ietk import metric
    # load an image
    dset = IDRiD('./data/IDRiD_segmentation')
    img, labels = dset['IDRiD_39']
    #  he = labels['HE']
    #  ma = labels['MA']
    #  ex = labels['EX']
    #  se = labels['SE']
    #  od = labels['OD']
    # set background pure black.
    #  bg = util.get_background(img)
    #  img[bg] = 0

    I = img.copy()
    #  I = img[1500:3000, 1500:3000, :]
    #  Imask = labels['MA'][1500:3000, 1500:3000]
    sh(I, 1)
    ## brighten dark areas
    z2 = brighten_dark_areas_v2(I)
    sh(z2, 2)
    #  z3 = brighten_dark_areas_v1(I)
    #  sh(z3, 3)
    #  assert np.allclose(z2, z3)
    import sys ; sys.exit()

    ## brighten bright areas
    z4 = brighten_bright_areas_v1(I)
    sh(z4, 4)
    #  z5 = brighten_bright_areas_v2(I)
    #  sh(z5, 5)
    #  assert np.allclose(z4, z5)

    ## darken bright areas
    z7 = darken_bright_areas_redgreen_v1(I)
    sh(z7, 7)
    #  #  z7[:,:,2] = 0

    ## darken the DARK areas
    z8 = darken_dark_areas_redgreen_v2(I)
    #  #  z8[:,:,2]
    sh(z8, 8)
    z9 = darken_dark_areas_redgreen_v1(I)
    sh(z9, 9)
    #  assert np.allclose(z8, z9)

    ## compose both brighten methods (b) and both darken methods (d)
    #  z20 = compose_brighten(I)
    #  sh(z20, 20)
    #  z21 = compose_darken(I)
    #  sh(z21, 21)
    #  z22 = compose_bd(I)
    #  sh(z22, 22)
    #  z23 = compose_db(I)
    #  sh(z23, 23)
    #  plt.show()

    ##  compose brighten dark areas and darken bright areas
    bd = compose_bd_simpler(I)
    #  bd = compose_bd(I)
    db = compose_db_simpler(I)
    #  db = compose_db(I)
    avged = bd/2 + db/2
    avged[bg] = 0
    #  avged2 = np.sqrt(bd*db)
    #  sh(bd, 30)
    #  sh(db, 31)
    #  sh(np.sqrt(bd*db), 32)
    sh(avged, 10)
    plt.figure() ; plt.imshow(avged)
    #  import sys ; sys.exit()

    ## evaluate
    print('ill')
    #  ill = methods.illuminate_dcp(I, focus_region=util.get_foreground(I))
    #  illsh = methods.illuminate_sharpen(I, focus_region=util.get_foreground(I))
    #  sh(illsh, 12)
    composedsharp = methods.sharpen(avged, focus_region=util.get_foreground(I))
    sh(composedsharp, 11)
    print('metric')
    for labelname, Imask in labels.items():
        print(labelname)
    #      for z in [I, bd, db, avged, ill, illsh, composedsharp]:
        for z in [I, illsh, composedsharp]:
        #  for z in [composedsharp]:
            print(metric.ks_test_max_per_channel(z, Imask, np.ones_like(I, dtype='bool')))
