from inputFilter import getImage, getLabel
from ui.file_dialog import save_model_file, select_model_file, labeled_picker
# bery vasic for now

imageSizeDefault = 128
def outputImg():
    raw = input('Enter the list: ')
    return getImage(raw)
def outputLabel():
    x = outputImg()
    return getLabel(x)
# window picker(picks pos & neg files respectively)
def outputImgP(parent=None):
    paths, labels = labeled_picker(parent=parent)
    return paths, labels
def savePath(parent=None):
    return save_model_file(parent=parent)
def loadPath(parent=None):
    return select_model_file(parent=parent)