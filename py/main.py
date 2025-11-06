import importlib, sys
def main():
    print("py script started with arg: ")
    input("press enter to exit..")
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip()
    else:
        mode = input(
            "--susieML interface--\n"
            "1 = Train (dense susieML)\n"
            "2 = Find Sussy(dense)\n"
            "3 = Train (CNN susieML)\n"
            "4 = Find Sussy(cnn)\n"
            "pick mode: "
        ).strip()
    # dense-type 
    if mode in ["1", '2']:
        dense = importlib.import_module("denseML")
        print("\nrunning denseML(NOT RECOMMENDED)")
        dense.main(mode)
    # cnn-type
    elif mode in ['3', '4']:
        cnn = importlib.import_module('torchML')
        print('\nrunning CNN type(better)')
        cnn.main(mode)
    else: print("\nInvalid selection. susie out")
    
if __name__ == "__main__":
    main()