import pandas as pd
import numpy as np
import xgboost as xgb
from skopt import space
from sklearn import svm
import deepchem.feat 
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

MOLWT_LOWER = 100
MOLWT_UPPER = 900
LOGP_LOWER = -4
LOGP_UPPER = 10


code3 = [
'Alpha-1A adrenergic receptor',
'Alpha-1B adrenergic receptor',
'Alpha-1D adrenergic receptor',
'Alpha-2 adrenergic receptor',
'Alpha-2A adrenergic receptor',
'Alpha-2B adrenergic receptor',
'Alpha-2C adrenergic receptor',
'Alpha-2Da adrenergic receptor',
'Alpha-2Db adrenergic receptor'
]


code4 =[
'Type-1 angiotensin II receptor',
'Type-1 angiotensin II receptor A',
'Type-1 angiotensin II receptor B',
'Type-1A angiotensin II receptor',
'Type-1B angiotensin II receptor',
'Type-2 angiotensin II receptor'
]


code7 = [
'B1 bradykinin receptor',
'B2 bradykinin receptor'
]


code8 = [
'Beta-1 adrenergic receptor',
'Beta-2 adrenergic receptor',
'Beta-3 adrenergic receptor',
'Beta-4C adrenergic receptor',
]


code21=[
'Dopamine receptor 1',
'Dopamine receptor 2',
'Dopamine receptor 3',
'Dopamine receptor 4',
'Dopamine D2-like receptor',
'D(1A) dopamine receptor',
'D(1B) dopamine receptor',
'D(1C) dopamine receptor',
'D(2) dopamine receptor',
'D(2) dopamine receptor A',
'D(2) dopamine receptor B',
'D(3) dopamine receptor',
'D(4) dopamine receptor',
'D(1)-like dopamine receptor',
'D(2)-like dopamine receptor',
'D(5)-like dopamine receptor'
]


code58 = [
'Mu-type opioid receptor',
'Kappa-type opioid receptor',
'Delta-type opioid receptor'
]


coded_gpcr3 = 'Alpha-adrenergic receptor'
coded_gpcr4 = 'Angiotensin receptor'
coded_gpcr7 = 'B-bradykinin receptor'
coded_gpcr8 = 'Beta-adrenergic receptor'
coded_gpcr21 = 'Dopamine receptor'
coded_gpcr58 = 'Opioid receptor'

names = [code3, code4, code7, code8, code21, code58]
coded_gpcrs = [coded_gpcr3, coded_gpcr4, coded_gpcr7, coded_gpcr8, coded_gpcr21, coded_gpcr58]

rdkit_desc = [
            'MinAbsPartialCharge',
            'NumRadicalElectrons',
            'HeavyAtomMolWt',
            'MaxAbsEStateIndex',
            'MaxAbsPartialCharge',
            'MaxEStateIndex',
            'MinPartialCharge',
            'ExactMolWt',
            'MolWt',
            'NumValenceElectrons',
            'MinEStateIndex',
            'MinAbsEStateIndex',
            'MaxPartialCharge',
            'BalabanJ',
            'BertzCT',
            'Chi0',
            'Chi0n',
            'Chi0v',
            'Chi1',
            'Chi1n',
            'Chi1v',
            'Chi2n',
            'Chi2v',
            'Chi3n',
            'Chi3v',
            'Chi4n',
            'Chi4v',
            'HallKierAlpha',
            'Ipc',
            'Kappa1',
            'Kappa2',
            'Kappa3',
            'LabuteASA',
            'PEOE_VSA1',
            'PEOE_VSA10',
            'PEOE_VSA11',
            'PEOE_VSA12',
            'PEOE_VSA13',
            'PEOE_VSA14',
            'PEOE_VSA2',
            'PEOE_VSA3',
            'PEOE_VSA4',
            'PEOE_VSA5',
            'PEOE_VSA6',
            'PEOE_VSA7',
            'PEOE_VSA8',
            'PEOE_VSA9',
            'SMR_VSA1',
            'SMR_VSA10',
            'SMR_VSA2',
            'SMR_VSA3',
            'SMR_VSA4',
            'SMR_VSA5',
            'SMR_VSA6',
            'SMR_VSA7',
            'SMR_VSA8',
            'SMR_VSA9',
            'SlogP_VSA1',
            'SlogP_VSA10',
            'SlogP_VSA11',
            'SlogP_VSA12',
            'SlogP_VSA2',
            'SlogP_VSA3',
            'SlogP_VSA4',
            'SlogP_VSA5',
            'SlogP_VSA6',
            'SlogP_VSA7',
            'SlogP_VSA8',
            'SlogP_VSA9',
            'TPSA',
            'EState_VSA1',
            'EState_VSA10',
            'EState_VSA11',
            'EState_VSA2',
            'EState_VSA3',
            'EState_VSA4',
            'EState_VSA5',
            'EState_VSA6',
            'EState_VSA7',
            'EState_VSA8',
            'EState_VSA9',
            'VSA_EState1',
            'VSA_EState10',
            'VSA_EState2',
            'VSA_EState3',
            'VSA_EState4',
            'VSA_EState5',
            'VSA_EState6',
            'VSA_EState7',
            'VSA_EState8',
            'VSA_EState9',
            'FractionCSP3',
            'HeavyAtomCount',
            'NHOHCount',
            'NOCount',
            'NumAliphaticCarbocycles',
            'NumAliphaticHeterocycles',
            'NumAliphaticRings',
            'NumAromaticCarbocycles',
            'NumAromaticHeterocycles',
            'NumAromaticRings',
            'NumHAcceptors',
            'NumHDonors',
            'NumHeteroatoms',
            'NumRotatableBonds',
            'NumSaturatedCarbocycles',
            'NumSaturatedHeterocycles',
            'NumSaturatedRings',
            'RingCount',
            'MolLogP',
            'MolMR'
]


search_params = {
    
    'grids':
        [
            {
                "n_estimators": [100,1000,5000],
                "min_samples_leaf": [3, 5, 10],
                'criterion': ['gini', 'entropy']
            },
        
            {
                "kernel":['rbf','linear'], 
                "C":[1,10,100,1000]
            },
            
            {
                "hidden_layer_sizes": [100, 1000, 2000, 5000], 
                "max_iter": [100, 1000, 5000],
                "alpha": [0.1,0.01,0.001]
            },
            
            {
                "n_estimators": [100, 1000, 5000],
                "learning_rate": [0.1, 0.01, 0.001],
                "max_depth": [3, 5, 10],
                'objective':['multi:softprob']
            }
        ],
    
    'bayes':
        [

            {
                'n_estimators': space.Integer(10, 1000),  
                'min_samples_leaf': space.Real(0.01, 10, prior='uniform'), 
                'criterion': space.Categorical(['gini', 'entropy'])
            },
        
            {
                'C': space.Real(1e-3, 1e3),
                'kernel': space.Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
                'degree': space.Integer(1, 100),
            },
            
    
            {
                'alpha': space.Real(0.0001, 0.05),
                'hidden_layer_sizes': space.Integer(100, 5000),
                'activation': space.Categorical(['tanh', 'relu']),
            }
        ]
            
}


learners = {
    'RF':
        RandomForestClassifier,
    'SVM':
        svm.SVC,
    'MLP':
        MLPClassifier,
    'XGB':
        xgb.XGBClassifier,
}


ckp_files = {
    
    'RF':
        'rf_ckp.out',
    'SVM':
        'svm_ckp.out',
    'MLP':
        'mlp_ckp.out',
    'XGB':
        'xgb_ckp.out',
    'DNN':
        'dnn_ckp.out'
}

pkl_files = {
        'RF':
            'rforest.pkl', 
        'SVM':
            'svm.pkl', 
        'MLP':
            'mlp.pkl',
        'XGB':
            'xgb.pkl',
        'DNN':
            'dnn.pkl'
    }


if __name__ == '__main__':


    # gpcr_ligand_df = pd.read_csv('gpcr_ligand_.csv')
    # rdkit_featurizer = deepchem.feat.RDKitDescriptors()

    # features = rdkit_featurizer.featurize('CC1CN(CC(N1)C)CC2=CC=C(C=C2)C3=CC=CC=C3CN(CCC4=CC(=CC=C4)OC)C(=O)NC5CCCCC5')
    # features = features.reshape((208,))
    # desc_names = [rdkit_featurizer.descList[i][0] for i in range(len(rdkit_featurizer.descList))]
    # columns = ['smiles'] + desc_names + ['labels']

    # labels = list(gpcr_ligand_df['gpcr_binding_encoded'])
    # smiles = list(gpcr_ligand_df['canonical_smiles'])
    # desc_df = pd.DataFrame(columns=columns)
    

    # for i in range(len(smiles)):
    #     features = rdkit_featurizer.featurize(smiles[i])
    #     features = list(features.reshape((208,)))
        
    #     row = [smiles[i]] + features + [labels[i]]
    #     desc_df.loc[len(desc_df)] = row

    # print(desc_df)
    
    print(MLPClassifier().n_layers_)