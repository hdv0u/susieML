import os
from tkinter import Tk, filedialog
def labeledPicker():
    root = Tk()
    root.withdraw()
    # positive(Susie files)
    pos = filedialog.askopenfilenames(
        title="get sussy (positive) images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    # negative(any excluding Susie)
    neg = filedialog.askopenfilenames(
        title="get non-sussy (negative) images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    # combine n label
    paths = list(pos) + list(neg)
    labels = [1] * len(pos) + [0] * len(neg)
    return paths, labels

# save model thing
def pick_save(default_name='model', default_ext='.pt',targetfolder='models'):
    root = Tk()
    root.withdraw()
    
    os.makedirs(targetfolder, exist_ok=True)
    default_path = os.path.join(targetfolder, default_name + default_ext)
    
    save = filedialog.asksaveasfilename(
        title="Save model twan",
        defaultextension = default_ext,
        initialdir = targetfolder,
        initialfile = default_name,
        filetypes=[
            ('PyTorch model','*.pt'),
            ('PyTorch alt model','*.pth'),
            ('All files','*.*')
            ]
    )
    if not save: return None
    if not save.endswith(default_ext):
        save += default_ext

    return save