from termcolor import colored
import json

def dictcmp(dict1, dict2):
    diffkeys = [k for k in dict1 if dict1[k] != dict2[k]]
    for k in diffkeys:
        print(colored(f"|{k}|", 'blue'), f"{dict1[k]} ---> {dict2[k]}")


def get_config_from_json(json_file):
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            return config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)

def configcmp(config_path1, config_path2):
    config_dict1 = get_config_from_json(config_path1)
    config_dict2 = get_config_from_json(config_path2)
    dictcmp(config_dict1, config_dict2)

if __name__== '__main__':
    config_path1 = "/home/simonzhang/Specific/Modified/Neighbor2Neighbor/results/1634271033/config.json"
    config_path2 = "/home/simonzhang/Specific/Modified/Neighbor2Neighbor/results/1634439534/config_Neigh2Neigh_1634439534.json"
    configcmp(config_path1, config_path2)
