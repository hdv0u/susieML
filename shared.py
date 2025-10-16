from inputFilter import getImage, getLabel
from imagePicker import labeledPicker, pick_save
# bery vasic for now

imageSizeDefault = 128
def outputImg():
    raw = input('Enter the list: ')
    return getImage(raw)
def outputLabel():
    x = outputImg()
    return getLabel(x)
# window picker(picks pos & neg files respectively)
def outputImgP():
    paths, labels = labeledPicker()
    x = getImage(' '.join(paths))
    return x, labels
def savePath():
    return pick_save()