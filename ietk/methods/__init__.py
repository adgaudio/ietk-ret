import re
from dataclasses import dataclass as _dataclass

from .brighten_darken_iciar2020 import brighten_darken
from .sharpen_img import sharpen


@_dataclass
class _bd:
    method_name: str
    def __call__(self, img, fg):
        return brighten_darken(
            img=img, method_name=self.method_name, focus_region=fg)
    def __repr__(self):
        return f'NAME{self.method_name}(img, focus_region)'
    def __str__(self):
        return self.__repr__()


def identity(img, focus_region, **kwargs):
    return img


class _all_methods(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['identity'] = identity

    def __getitem__(self, method_name):
        if re.match(r'(avg\d+:)?s?[ABCDWXYZ](\+s?[ABCDWXYZ])*$', method_name):
            if method_name.startswith('avg'):
                method_name = method_name.split(':', 1)[1]
            # brighten darken methods from ICIAR2020 paper
            return _bd(method_name)
        else:
            return Exception("Unrecognized method name: %s" % method_name)


all_methods = _all_methods()
