import importlib, sys

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

def run_mode(mode, log_fn=print, frame_fn=None, progress_fn=None, stop_fn=None, 
             parent=None, model_save=None, train_paths=None, train_labels=None, 
             load_model=None, multi_class=False, label_widget=None):
    module_name = MODE_REGISTRY.get(mode)
    if not module_name:
        log_fn("invalid mode")
        return
    
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        log_fn(f"failed to load model '{module_name}': {e}")
        return
    
    if not hasattr(mod, "main"):
        log_fn(f"{module_name} has no main()")
        return
    mod.main(
        mode, 
        log_fn=log_fn, 
        progress_fn=progress_fn, 
        stop_fn=stop_fn, 
        parent=parent, 
        model_save=model_save, 
        train_paths=train_paths,
        train_labels=train_labels,
        load_model=load_model,
        multi_class=multi_class,
        label_widget=label_widget,
        frame_fn=frame_fn
        )

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
    app = QApplication(sys.argv)
    if not app:
        app = QApplication.instance()
    
    from ui.main_window import TestUI
    window = TestUI()
    window.show()
    sys.exit(app.exec_())

def main():
    if "--cli" in sys.argv:
        cli()
    else:
        try:
            gui()
        except (ImportError, RuntimeError,OSError) as e:
            print("GUI failed, fallback to CLI")
            print(e)
            cli()
    
if __name__ == "__main__":
    main()