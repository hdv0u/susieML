import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
subparsers = parser.add_subparsers(dest='command')
# Subcommand 1
parser_train = subparsers.add_parser('train', help='Train a model')
parser_train.add_argument('-i', type=str, help='Model to train')
parser_train.add_argument('-d', type=int, help='Model depth', default=3)
# preproc sections
parser_train.add_argument('-br', type=int, help='brightness adjustment', default=0)
parser_train.add_argument('-r', type=float, help='rotation range', default=0.0)
parser_train.add_argument('-gn', type=float, help='gaussian noise stddev', default=0.0)
parser_train.add_argument('--multi', action='store_true', help='Use multi-class training')

# Subcommand 2
parser_detect = subparsers.add_parser('detect', help='Detect using a model')
parser_detect.add_argument('-i', type=str, help='Model to use for detection')
parser_detect.add_argument('-d', type=int, help='Model depth', default=3)
# inference parameters
parser_detect.add_argument('-t', type=float, help='Detection threshold', default=0.5)
parser_detect.add_argument('-sl', type=int, help='Side length for detection', default=128)
parser_detect.add_argument('-st', type=int, help='Step size for detection', default=96)

from core.frame_sinks import opencv_sink
from core.frame_sources import screen_source
from core.model.convnn_runner import CNNTrainer, CNNInference
# work in progress