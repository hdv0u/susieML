def convert_linear_to_fcn(old_model, new_model):
    linear = old_model.linear
    
    out_channels =linear.out_features
    in_features = linear.in_features
    
    conv_w = linear.weight.view(out_channels, 256, 4, 4)
    conv_b = linear.bias
    
    new_model.classifier.weight.data = conv_w
    new_model.classifier.bias.data = conv_b
    
    return new_model