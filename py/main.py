import importlib, sys
def main():
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
    # dense-type 
    if mode in ["1", '2']:
        dense = importlib.import_module("densenn")
        print("\nrunning densenn (v1.0)")
        dense.main(mode)
    # cnn-type
    elif mode in ['3', '4']:
        cnn = importlib.import_module('convnn')
        print('\nrunning classic CNN(recommended) (v1.1)')
        cnn.main(mode)
        
    elif mode in ['5', '6']:
        cnn = importlib.import_module('resnet1')
        print(f'\nrunning resNet(DEEP) (v1.2)')
        cnn.main(mode)
    else: print("\nInvalid selection. susie out")
    
if __name__ == "__main__":
    main()