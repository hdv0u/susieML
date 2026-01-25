from PyQt5.QtWidgets import QFileDialog

def select_model_file(parent=None, default_folder='models',default_ext='.pt'):
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Load model",
        default_folder,
        f"PyTorch Model (*{default_ext} *.pth);;NPZ Files (*.npz);;All Files (*)"
    )
    return file_path or None

def save_model_file(parent=None, default_name='model', default_folder='models', default_ext='.pt'):
    from os import makedirs, path
    makedirs(default_folder, exist_ok=True)
    default_path = path.join(default_folder, default_name + default_ext)
    
    file_path, _ = QFileDialog.getSaveFileName(
        parent,
        "Save model twan",
        default_path,
        f"PyTorch Model(*{default_ext} *.pth);;NPZ Files (*.npz);;All Files (*)"
    )
    if not file_path:
        return None
    if not file_path.endswith(default_ext):
        file_path += default_ext
    return file_path

def labeled_picker(parent=None):
    pos_files, _ = QFileDialog.getOpenFileNames(
        parent,
        "Get Sussy (positive images)",
        "",
        "Images (*.jpg *.jpeg *.png *.bmp *.webp *.gif)"
    )
    neg_files, _ = QFileDialog.getOpenFileNames(
        parent,
        "Get Non-Sussy (negative images)",
        "",
        "Images (*.jpg *.jpeg *.png *.bmp *.webp *.gif)"
    )
    paths = pos_files + neg_files
    labels = [1]*len(pos_files) + [0]*len(neg_files)
    return paths, labels

def labeled_picker_multi(parent=None, class_names=None):
    if class_names is None:
        class_names = ["C0", "C1", "C2"]
    
    class_map = {name: idx for idx, name in enumerate(class_names)}
    all_paths = []
    all_labels = []
    
    for name in class_names:
        files, _ = QFileDialog.getOpenFileNames(
            parent,
            f"Select images for {name}",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.gif)"
        )
        labels = [class_map[name]] * len(files)
        all_paths.extend(files)
        all_labels.extend(labels)
        
    return all_paths, all_labels