import os

def _get_qfiledialog():
    try:
        from PyQt5.QtWidgets import QFileDialog, QApplication
        if QApplication.instance() is None:
            return None
        return QFileDialog
    except Exception:
        return None

def _input_path(prompt):
    try:
        value = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return value or None

def select_model_file(parent=None, default_folder='models', default_ext='.pt'):
    QFileDialog = _get_qfiledialog()
    if QFileDialog is not None:
        file_path, _ = QFileDialog.getOpenFileName(
            parent,
            "Load model",
            default_folder,
            f"PyTorch Model (*{default_ext} *.pth);;NPZ Files (*.npz);;All Files (*)"
        )
        return file_path or None

    prompt = f"Model file path [{default_folder}]: "
    return _input_path(prompt)

def save_model_file(parent=None, default_name='model', default_folder='models', default_ext='.pt'):
    os.makedirs(default_folder, exist_ok=True)
    default_path = os.path.join(default_folder, default_name + default_ext)
    QFileDialog = _get_qfiledialog()
    if QFileDialog is not None:
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

    prompt = f"Save model path [{default_path}]: "
    file_path = _input_path(prompt) or default_path
    if not file_path.endswith(default_ext):
        file_path += default_ext
    return file_path

def labeled_picker(parent=None):
    QFileDialog = _get_qfiledialog()
    if QFileDialog is not None:
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
        labels = [1] * len(pos_files) + [0] * len(neg_files)
        return paths, labels

    print("Enter positive image paths, one per line. Blank line to finish.")
    pos_files = []
    while True:
        path = _input_path("positive> ")
        if not path:
            break
        pos_files.append(path)

    print("Enter negative image paths, one per line. Blank line to finish.")
    neg_files = []
    while True:
        path = _input_path("negative> ")
        if not path:
            break
        neg_files.append(path)

    paths = pos_files + neg_files
    labels = [1] * len(pos_files) + [0] * len(neg_files)
    return paths, labels

def labeled_picker_multi(parent=None, class_names=None):
    if class_names is None:
        class_names = ["C0", "C1", "C2"]

    QFileDialog = _get_qfiledialog()
    if QFileDialog is not None:
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
    
    # fallback to CLI input
    class_map = {name: idx for idx, name in enumerate(class_names)}
    all_paths = []
    all_labels = []
    print("Enter image paths for each class. Blank line to move to the next class.")
    for name in class_names:
        print(f"Class '{name}':")
        while True:
            path = _input_path(f"{name}> ")
            if not path:
                break
            all_paths.append(path)
            all_labels.append(class_map[name])

    return all_paths, all_labels
