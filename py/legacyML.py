# susieML NumPy edition``
# the plan is to add torch and tensorflow as backup ML shi
# goal is to expand image set instead of manually writing the file PATH
import numpy as np
from shared import outputImg, outputLabel, outputImgP
from preproc import screen_rec, image_proc, new_augment

# video = screen_rec(size=(64,64))
train_path, train_labels = outputImgP()
print(train_path, train_labels)
test_path = [
    'c:/Users/Dave/susieML/susiedata/test/actualtest.jpg',
]
test_labels = [[1]]
debugSwitch = True
# debug = [image_proc(i, debug=debugSwitch) for i in train_path] # remove hash if debug
train_input, checker_train = new_augment(train_path, train_labels, augment_count=10, debug=debugSwitch)
train_input = train_input.reshape(train_input.shape[0], -1)
test_input = np.array([image_proc(p, augment=False, debug=debugSwitch) for p in test_path]).reshape(len(test_path), -1)
checker_test = np.array(test_labels).reshape(len(test_path), 1)

input_size = train_input.shape[1]
hidden_size = 32
output_size = 1
weight1 = np.random.randn(input_size, hidden_size) * 0.01
bias1 = np.zeros((hidden_size,))
weight2 = np.random.randn(hidden_size, output_size) * 0.01
bias2 = np.zeros((output_size,))

learn, gen = 0.01, 1000
epsilon= 1e-9 # idk why but its there

# goal is to integrate video with class or repetitive code

# train and learn(images)
for generation in range(gen):
    # forward pass or idk
    hidden = np.maximum(0, train_input @ weight1 + bias1)
    z = hidden @ weight2 + bias2
    pred = 1 / (1 + np.exp(-z))
    # BCE cuz only Susie(yes/no)
    loss = -np.mean(
        checker_train * np.log(pred + epsilon) 
        + (1 - checker_train) * np.log(1 - pred + epsilon)
        )
    # rest is improvements and fixes(training part)
    error = pred - checker_train
    grad_weight2 = hidden.T @ error / train_input.shape[0]
    grad_bias2 = np.mean(error, axis=0)
    
    d_hidden = (error @ weight2.T) * (hidden > 0) # relu deriv
    grad_weight1 = train_input.T @ d_hidden / train_input.shape[0]
    grad_bias1 = np.mean(d_hidden, axis=0)
    
    weight2 -= learn * grad_weight2
    bias2 -= learn * grad_bias2
    weight1 -= learn * grad_weight1    
    bias1 -= learn * grad_bias1
        
    if generation % 100 == 0:
        hidden_test = np.maximum(0, test_input @ weight1 + bias1)
        z_test = hidden_test @ weight2 + bias2
        pred_test = 1 / (1 + np.exp(-z_test))
            
        # loss for test_path input
        test_loss = -np.mean(
            checker_test * np.log(pred_test + epsilon) 
            + (1 - checker_test) * np.log(1 - pred_test + epsilon)
            )
        print(f"gen:{generation}, train_path loss:{loss:.5f}, test_path loss:{test_loss:.5f}")
        
print('train_input shape:', train_input.shape)
print('checeker_train shape:', checker_train.shape)
print('z shape:', z.shape)
print('loss shape:', loss.shape)
print('gradw1 shape:', grad_weight1.shape)
print('gradw2 shape:', grad_weight2.shape)
