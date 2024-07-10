import os

from omegaconf import OmegaConf, DictConfig, ListConfig

here = os.path.abspath(os.path.dirname(__file__))


def setup_env():
    bash_script = os.path.join(here, '../setup.sh')
    os.system('bash ' + bash_script)


def prefix_dict(p, d, sep='_'):
    return {f'{p}{sep}{k}': v for k, v in d.items()}


def flatten_dict(d):
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = prefix_dict(k, flatten_dict(v), sep='.')
            res.update(v)
        else:
            res[k] = v
    return res


def dict2commandline(d, item_delimiter='|'):
    res = []
    for k, v in flatten_dict(d).items():
        res.append(f'{k}={v if v is not None else "null"}')
    return item_delimiter.join(res)


def str2dict(str, item_delimiter='|'):
    return _recursive_to_dict(OmegaConf.from_dotlist(str.split(item_delimiter)))


def _recursive_to_dict(d):
    res = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            v = _recursive_to_dict(v)
        elif isinstance(v, ListConfig):
            v = list(v)
        res[k] = v
    return res
