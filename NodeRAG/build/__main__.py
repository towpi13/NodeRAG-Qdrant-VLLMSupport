import argparse
import yaml
import os

from .Node import NodeRag,NodeConfig

parser = argparse.ArgumentParser(description='TGRAG Build')
parser.add_argument('-f','--folder_path', type=str, help='The folder path of the document')


args = parser.parse_args()

config_path = os.path.join(args.folder_path,'Node_config.yaml')
if not os.path.exists(config_path):
    config = NodeConfig.from_main_folder(args.folder_path)
    print("Please modify the config file and run the command again")
    exit(0)
else:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = NodeConfig(config)
    




ng = NodeRag(config)
ng.run()









