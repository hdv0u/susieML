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