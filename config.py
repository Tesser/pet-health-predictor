import random
random.seed(7)

config = {
    'model_path' : "./model/",
    'scaler_path' : "./utils/",
    'ohe_path' : "./utils/",
    "split_seed" : 255,
    'test_split' : 0.2,
    'model_seed' : 8,
    'disease_list' : ['건강', '간질환', '신장질환','심장질환','종양'],
    'score_list': ['skin',
                   'joint', 'heart', 'eye', 'respi', 'oral', 'diget', 'urin', 'brain',
                   'hormon']
}