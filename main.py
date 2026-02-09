from argparse import ArgumentParser
from configs.utils import get_config
from dataset.data_utils import Preprocessor
from tasks import train, test

parser = ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)
args = parser.parse_args()
config_file = args.config_file

if __name__ == "__main__":
    config = get_config(config_file)
    Preprocessor(config.dataset).run()
    train(config)
    test(config)

