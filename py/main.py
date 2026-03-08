import importlib, sys, os, traceback
from core.frame_sinks import opencv_sink

if getattr(sys, 'frozen', False):
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(__file__)

# Paths to folders
UI_PATH = os.path.join(BASE_PATH, "ui")
CORE_PATH = os.path.join(BASE_PATH, "core")
MODELS_PATH = os.path.join(BASE_PATH, "models")  # if you create this later
IMAGES_PATH = os.path.join(BASE_PATH, "images")  # any icons/images
CONFIG_PATH = os.path.join(BASE_PATH, "configs")

def log(msg, to_file=True):
    print(msg)
    if to_file:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

# correspond the num with the filename of a new model
# it should be:
# {odd}: train x
# {even}: detect x
MODE_REGISTRY = {
    "1": "densenn",
    "2": "densenn",
    "3": "convnn",
    "4": "convnn",
    "5": "resnet1",
    "6": "resnet1",
}

DEBUG = '--debug' in sys.argv

def run_mode(mode, log_fn=print, frame_fn=None, progress_fn=None, stop_fn=None, 
             parent=None, model_save=None, train_paths=None, train_labels=None, 
             load_model=None, multi_class=False, label_widget=None, debug=False):
    module_name = MODE_REGISTRY.get(mode)
    if not module_name:
        log_fn("invalid mode")
        return
    if debug:
        log_fn(f"debug: running mode {mode}")
    if frame_fn is None:
        frame_fn = opencv_sink()
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        log_fn(f"failed to load model '{module_name}': {e}")
        log_fn(traceback.format_exc())
        return
    
    if not hasattr(mod, "main"):
        log_fn(f"{module_name} has no main()")
        return
    try:
        mod.main(
            mode, 
            log_fn=log_fn, 
            progress_fn=progress_fn, 
            stop_fn=stop_fn, 
            frame_fn=frame_fn,
            parent=parent, 
            model_save=model_save, 
            train_paths=train_paths,
            train_labels=train_labels,
            load_model=load_model,
            multi_class=multi_class,
        )
    except Exception as e:
        log_fn(f"Exception in {module_name}.main: {e}")
        log_fn(traceback.format_exc())
# fixing cli soon
def cli():    
    try:
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
        run_mode(mode, debug=DEBUG)
    except Exception as e:
        log(f"CLI Exception: {e}")
        log(traceback.format_exc())
def gui():
    from PyQt5.QtWidgets import QApplication
    try:
        app = QApplication(sys.argv)
        if not app:
            app = QApplication.instance()
        
        from ui.window_test import TestWindow
        window = TestWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        log(f"GUI Exception: {e}")
        log(traceback.format_exc())
        raise  # fallback handled in main()

def main():
    if "--cli" in sys.argv:
        cli()
    else:
        try:
            gui()
        except Exception as e:
            log("GUI failed, fallback to CLI")
            log(str(e))
            log(traceback.format_exc())
            cli()
    
if __name__ == "__main__":
    main()