from inputFilter import getImage, getLabel
from imagePicker import labeledPicker
# bery vasic for now
# copy paste
imageSizeDefault = 128
def outputImg():
    raw = input('Enter the list: ')
    return getImage(raw)
def outputLabel():
    x = outputImg()
    return getLabel(x)
# window picker
def outputImgP():
    paths, labels = labeledPicker()
    x = getImage(' '.join(paths))
    return x, labels