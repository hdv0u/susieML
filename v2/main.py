# should be the entry point but still wip
import sys
from PyQt5.QtWidgets import QApplication

from ui.window import TestWindow
from ui.mediator import Mediator
from config import cfg
def main():
    app = QApplication(sys.argv)
    ui = TestWindow()
    mediator = Mediator(cfg, ui)
    ui.set_mediator(mediator)
    ui.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()