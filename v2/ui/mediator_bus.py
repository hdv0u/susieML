from PyQt5.QtCore import QObject, pyqtSignal

class UIBus(QObject):
    train_requested = pyqtSignal(object) # dataset
    run_requested = pyqtSignal(str) # model
    stop_requested = pyqtSignal()
    
bus = UIBus()
    