import numpy as np


def make_str_to_float(json):
    json = {k: np.nan if not v else v for k, v in json.items()}
    for keys in json.keys():
        try:
            json[keys] = float(json[keys])
        except:
            pass
    return json

