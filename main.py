import importlib
# modes duh
mode = input(
    "1 = Train (dense susieML)\n"
    "2 = Find Sussy(dense)\n"
    "3 = Train (torch susieCNN)\n"
    "4 = Find Sussy(torch susieCNN...)\n"
    "pick mode: "
    ).strip()
if mode in ["1", '2']:
    dense = importlib.import_module("denseML")
    print("running denseML(NOT RECOMMENDED)")
    dense.main(mode)
elif mode in ['3', '4']:
    torchML = importlib.import_module('torchML')
    print('running CNN torch(better way)')
    torchML.main(mode)
else: print("Invalid selection. susie out")