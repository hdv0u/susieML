import importlib, sys
MODE_REGISTRY = {
    "1": "densenn",
    "2": "densenn",
    "3": "convnn",
    "4": "convnn",
    "5": "resnet1",
    "6": "resnet1",
}

def run_mode(mode, log_fn=print, frame_fn=None, progress_fn=None, stop_fn=None):
    module_name = MODE_REGISTRY.get(mode)
    if not module_name:
        log_fn("invalid mode")
        return
    
    mod = importlib.import_module(module_name)
    mod.main(mode, log_fn=log_fn, progress_fn=progress_fn, stop_fn=stop_fn)

def cli():    
    mode = input(
        "--susieML interface--\n"
        "1 = Train (dense susieML)\n"
        "2 = Find Sussy(dense)\n"
        "3 = Train (CNN susieML)\n"
        "4 = Find Sussy(cnn)\n"
        "5 = Train (ResNet CNN)\n"
        "6 = Find Sussy(resnet)\n"
        "pick mode: "
    ).strip()
    
    run_mode(mode)
    
def gui():
    from PyQt5.QtWidgets import QApplication
    from ui.main_window import TestUI
    
    app = QApplication(sys.argv)
    window = TestUI()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    if "--cli" in sys.argv:
        cli()
    else:
        try:
            gui()
        except Exception as e:
            print("GUI failed, fallback to CLI")
            print(e)
            cli()