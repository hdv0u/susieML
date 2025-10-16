from inputFilter import getImage, getLabel
from imagePicker import labeledPicker, save_model, load_model
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
    return save_model()
def loadPath():
    return load_model()