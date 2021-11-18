import numpy as np
import pandas as pd


def preprocessing_json(json):

    json = {k: np.nan if not v else v for k, v in json.items()}
    for keys in json.keys():
        try:
            json[keys] = float(json[keys])
        except:
            pass
    df_json = pd.DataFrame(json, index=['0'])
    df_json = df_json[['Hg', 'As', 'Cd', 'Pb', 'Al', 'Ba', 'U', 'Bi', 'Ni', 'Ti', 'Cs', 'Sb',
             'Ca', 'Mg', 'Na', 'K', 'Cu', 'Zn', 'P', 'Fe', 'Mn', 'Cr', 'Se', 'Co',
             'Li', 'V', 'Mo', 'B', 'Ca_Mg', 'Ca_P', 'Na_K', 'Zn_Cu', 'Na_Mg', 'Ca_K',
             'age', 'sex', 'weight', 'BCS', 'Glucose', 'BUN', 'Creatinine',
             'BUN_CREratio', 'ALP', 'ALT', 'T_protein', 'Albumin', 'Globulin',
             'A_G ratio', 'T_bilirubin', 'GGT', 'T_cholesterol', 'Phosphorus',
             'Calcium', 'AST_GOT', 'ALT_GPT', 'TG', 'RBC', 'HCT', 'HGB', 'MCV',
             'MCH', 'MCHC', 'RDW-CV', 'shap_RETIC', 'percent_RETIC', 'RETHGB', 'WBC',
             'percent_NEU', 'percent_LYM', 'percent_MON', 'percent_EO',
             'percent_BASO', 'percent_GRA', 'shap_NEU', 'shap_LYM', 'shap_MON',
             'shap_EO', 'shap_BASO', 'shap_GRA', 'PLT', 'MPV', 'PDW', 'THT', 'Na+',
             'K+', 'Ca++', 'Cl-', 'pH', 'PCO2', 'PO2', 'HCO3-', 'BEecf(Ven)', 'tCO2',
             'SO2', 'stO2', 'Hct(Ven)']]

    df_json.fillna(-999, inplace=True)

    return df_json

