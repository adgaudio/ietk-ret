import SimpleITK as sit


def tonp(img):
    """convenience method to convert sitk to numpy"""
    return sitk.GetArrayFromImage(img)


def fromnp(img):
    return sitk.GetImageFromArray(img)


def read_img(fp):
    reader = sitk.ImageFileReader()
    reader.SetFileName(fp)
    image = reader.Execute()
    np.
    return image


def save_img(fp, out):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(fp)
    writer.Execute(out)
