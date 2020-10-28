import yaml
import pprint

FILE = "dummy.yaml"

def get_yaml():
    """ reads a properly formatted yaml configuration file & parses it into python dictionary format  """
    with open(FILE, 'r') as configs:
        configs = yaml.load(configs, yaml.FullLoader)
    return configs

content = get_yaml()
# print(content)
pprint.pprint(content, indent=2, compact=True, width=5, sort_dicts=False)