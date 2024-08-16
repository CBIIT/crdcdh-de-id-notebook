import tarfile, json, re, requests, yaml, boto3, sys, csv
from datetime import datetime 

def untar_file(tar_file, dest_dir):
    """
    untar file to dest_dir
    """
    with tarfile.open(tar_file, 'r:gz') as _tar:            
        for member in _tar:
            if member.isdir():
                continue
            _tar.makefile(member, dest_dir + '/' + member.name)

def load_json(json_file):
    """
    load json file
    """
    with open(json_file, 'r') as f:
        return json.load(f)


def pretty_print_json(json_file):
    """
    print json file prettily
    """
    print(json.dumps(load_json(json_file), indent=4))

def get_date_time(format = "%Y-%m-%d-%H-%M-%S-%f"):
    """
    get current time in format
    """  
    return datetime.strftime(datetime.now(), format)[:-3]

# Function to preprocess text
def preprocess_text(text):
    text = text.lower() if text else ''  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text) if text else ''  # Remove punctuation
    return text

def get_boto3_session(aws_profile=None):
    return boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()

"""
download file from url and load into dict 
:param: url string
:return: value dict
"""
def download_file_to_dict(url):
    # NOTE the stream=True parameter below
    file_ext = url.split('.')[-1]
    with requests.get(url) as r:
        if r.status_code > 400: 
            raise Exception(f"Can't find model file at {url}, {r.content}!")
        if file_ext == "json":
            return r.json()
        elif file_ext in ["yml", "yaml"]: 
            return yaml.safe_load(r.content)
        else:
            raise Exception(f'File type is not supported: {file_ext}!')
        
"""
Dump list of dictionary to json file, caller needs handle exception.
:param: dict_list as list of dictionary
:param: file_path as str
:return: boolean
"""
def dump_dict_to_json(dict, path):
    if not dict or len(dict.items()) == 0:
        return False 
    output_file = open(path, 'w', encoding='utf-8')
    json.dump(dict, output_file) 
    return True

"""
Extract exception type name and message
:return: str
"""
def get_exception_msg():
    ex_type, ex_value, exc_traceback = sys.exc_info()
    return f'{ex_type.__name__}: {ex_value}'


"""
Load yaml file to dict
:param: yaml_file as str
:return: dict
"""
def yaml2dict(yaml_file):
    # Custom constructor to handle hexadecimal values
    def hex_constructor(loader, node):
        value = loader.construct_sequence(node, deep=True)
        # Convert each value in the list to integer
        return (int(x, 16) for x in value)
    # Register the custom constructor for sequences
    yaml.add_constructor('tag:yaml.org,2002:seq', hex_constructor)
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data

class HexInt(int):
    """A class to represent an integer in hexadecimal format in YAML."""
    def __str__(self):
        return f"0x{self:X}"

def convert_tuple_to_hex(data):
    """Convert a tuple of integers to a list of HexInt."""
    return [HexInt(item) for item in data]

def process_dict_tags(list, key):
    """Process the dictionary to convert tuples to hex format."""
    for entry in list:
        if key in entry:
            entry[key] = convert_tuple_to_hex(entry[key])
    return list

def hex_int_representer(dumper, data):
        """Represent HexInt as a scalar in YAML."""
        return dumper.represent_scalar('tag:yaml.org,2002:int', str(data))

def dict2yaml(dict, file_path):
    yaml.add_representer(HexInt, hex_int_representer)
    with open(file_path, 'w') as file:
            yaml.dump(dict, file, default_flow_style=False)

"""
Dump list of dictionary to TSV file, caller needs handle exception.
:param: dict_list as list of dictionary
:param: file_path as str
:return: boolean
"""
def dump_dict_to_tsv(dict_list, file_path):
    if not dict_list or len(dict_list) == 0:
        return False 
    keys = dict_list[0].keys()
    with open(file_path, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys, delimiter='\t')
        dict_writer.writeheader()
        dict_writer.writerows(dict_list) 
    return True

