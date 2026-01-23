from ui.file_dialog import labeled_picker_multi
from core.model.convnn_runner import CNNTrainer
from preproc import new_augment

# simulate class names and fake files (use real small image paths or create dummy arrays)
class_names = ["Positive","Neutral","Negative"]
# create a fake mapping to three small images on disk, repeat per class or use same images
# For a real test point to actual image files
p_pos = ["C:/Users/Dave/Desktop/input/newpos/firefox_3bbi0kfMan.png", "C:/Users/Dave/Desktop/input/newpos/firefox_3HA2BcyMKJ.png"]
p_neu = ["C:/Users/Dave/Desktop/input/test/actualtest.jpg"]
p_neg = ["C:/Users/Dave/Desktop/input/newneg/firefox_81DWgnOhiC.png", "C:/Users/Dave/Desktop/input/newneg/firefox_8evejmSW2b.png"]

paths = p_pos + p_neu + p_neg
labels = [0]*len(p_pos) + [1]*len(p_neu) + [2]*len(p_neg)

print("UNIT TEST initial labels", labels)
X, y = new_augment(paths, labels, augment_count=1, mode='cnn', label_mode='cce')
print("after augment y shape, dtype, unique:", y.shape, y.dtype, set(y.tolist()))
