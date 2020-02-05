import re
import numpy as np
import scipy.ndimage as ndi
from cv2.ximgproc import guidedFilter

from . import sharpen_img
from .brighten_darken import solvet, solveJ, gf

def ta(img):
    return solvet(1-img, 1)
def td(img):
    return 1-solvet(1-img, 1)
def tb(img):
    I = img.copy()
    I[:,:,2] = 1
    return solvet(I, 1)
def tc(img):
    I = img.copy()
    I[:,:,2] = 1
    return 1-solvet(I, 1)
def A(img):
    return solveJ(img, 0, ta(img))
def B(img):
    return solveJ(img, 0, tb(img))
def C(img):
    return solveJ(img, 0, tc(img))
def D(img):
    return solveJ(img, 0, td(img))
def W(img):
    return solveJ(img, 1, ta(img))
def X(img):
    return solveJ(img, 1, tb(img))
def Y(img):
    return solveJ(img, 1, tc(img))
def Z(img):
    return solveJ(img, 1, td(img))

all_methods = {
    'identity': lambda img, focus_region, **kwargs: img,
    'A': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, ta(img)), ~focus_region),
    'A2': lambda img, focus_region, **kwargs: solveJ(img, 0, ta(img)),
    'B': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, tb(img)), ~focus_region),
    'C': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, tc(img)), ~focus_region),
    'C2': lambda img, focus_region, **kwargs: sharpen_img.sharpen(sharpen_img.sharpen(solveJ(img, 0, tc(img)), ~focus_region), ~focus_region),
    'C3': lambda img, focus_region, **kwargs: sharpen_img.sharpen(sharpen_img.sharpen(sharpen_img.sharpen(solveJ(img, 0, tc(img)), ~focus_region), ~focus_region), ~focus_region),
    'D': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, td(img)), ~focus_region),
    'W': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 1, ta(img)), ~focus_region),
    'X': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 1, tb(img)), ~focus_region),
    'Y': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 1, tc(img)), ~focus_region),
    'Z': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 1, td(img)), ~focus_region),
    r'A+X': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, ta(img))/2 + solveJ(img, 1, tb(img))/2, ~focus_region),
    r'C+X': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, tc(img))/2 + solveJ(img, 1, tb(img))/2, ~focus_region),
    r'A+C': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, ta(img))/2 + solveJ(img, 0, tc(img))/2, ~focus_region),
    r'A+Z': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, ta(img))/2 + solveJ(img, 0, td(img))/2, ~focus_region),
    r'A+C+X': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, ta(img))/3 + solveJ(img, 0, tc(img))/3 + solveJ(img, 1, tb(img))/3, ~focus_region),
    r'A+C+X+Z': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, ta(img))/4 + solveJ(img, 0, tc(img))/4 + solveJ(img, 1, tb(img))/4 + solveJ(img, 1, td(img))/4, ~focus_region),
}

_m = all_methods
all_methods.update({
    'A+B': lambda img, focus_region: sharpen_img.sharpen(A(img)/2 + B(img)/2, ~focus_region),
    #  'sA+sB2': lambda img, focus_region: _m['A'](img=img,focus_region=focus_region)/2 + _m['B'](img=img,focus_region=focus_region)/2,
    'B+C': lambda img, focus_region: sharpen_img.sharpen(B(img)/2 + C(img)/2, ~focus_region),
    'B+X': lambda img, focus_region: sharpen_img.sharpen(B(img)/2 + X(img)/2, ~focus_region),
    'A+B+C': lambda img, focus_region: sharpen_img.sharpen(A(img)/3 + B(img)/3 + C(img)/3, ~focus_region),
    'A+B+X': lambda img, focus_region: sharpen_img.sharpen(A(img)/3 + B(img)/3 + X(img)/3, ~focus_region),
    'B+C+X': lambda img, focus_region: sharpen_img.sharpen(B(img)/3 + C(img)/3 + X(img)/3, ~focus_region),
    'A+B+C+X': lambda img, focus_region: sharpen_img.sharpen(A(img)/4 + B(img)/4 + C(img)/4 + X(img)/4, ~focus_region),
})

all_methods.update({
    'A+B+C+W+X': lambda img, focus_region: sharpen_img.sharpen(A(img)/5 + B(img)/5 + C(img)/5 + W(img)/5 + X(img)/5, ~focus_region),
})

def sharpen_each(method_name):
    names = method_name.split('+')
    methods = {'A': A, 'B': B, 'C': C, 'D': D, 'W': W, 'X': X, 'Y': Y, 'Z': Z}
    def _sharpen_each(img, focus_region):
        return sum(sharpen_img.sharpen(methods[N](img), ~focus_region)/len(names) for N in names)
    return '+'.join(f's{N}' for N in names), _sharpen_each

all_methods.update([
    sharpen_each(method_name)
    for method_name in all_methods if re.match(r'[ABCDWXYZ]\+', method_name)])
