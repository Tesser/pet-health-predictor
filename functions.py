import numpy as np



def astype_json(json):
    json = {k: np.nan if not v else v for k, v in json.items()}
    for keys in json.keys():
        try:
            json[keys] = float(json[keys])
        except:
            pass
    return json

