import time
import deepchem.feat 
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from configuration import rdkit_desc
from datetime import datetime

def timestamp():
    '''
    Records timestamp
    '''
    ts = time.time()
    date_time = datetime.fromtimestamp(ts)
    stamp_str = date_time.strftime("%d-%m-%Y, %H:%M:%S")
    return stamp_str


def computeFP(smiles, labels):
    '''
    Takes SMILES strings and uses RDKit to convert them to ECFP6 vectors, returning pd.Dataframe
    '''
    moldata = [Chem.MolFromSmiles(mol) for mol in smiles]
    fpdata=[]
    for i, mol in enumerate(moldata):
        ecfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)]
        fpdata += [ecfp6]
    fp_df = pd.DataFrame(data=fpdata, index=smiles)
    fp_df['labels'] = labels
    return fp_df

def computeDesc(smiles, labels):
    '''
    Takes SMILES strings and uses DeepChem to calculate 111 molecular descriptors
    '''
    featurizer = deepchem.feat.RDKitDescriptors()
    initiate = featurizer.featurize('C')
    columns = [name[0] for name in featurizer.descList if name[0] in rdkit_desc]
    allowedInd = [i for i, desc in enumerate(featurizer.descList) if desc[0] in rdkit_desc]
    desc_data = []
    for mol in smiles:
        features = featurizer.featurize(mol)
        allowedFeats = [feature for i, feature in enumerate(features[0]) if i in allowedInd]
        desc_data += [allowedFeats]
    descriptors_df = pd.DataFrame(data=desc_data,index=smiles,columns=columns)
    descriptors_df['labels'] = labels
    return descriptors_df

        
def kfold_validation(estimator,  output, x, y, ckp_file=None, partial=False):
    '''
    Performs 5-fold cross validation experiment of ML model
    '''
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)
    ckp = 0
    acc_lst, f1_lst, mcc_lst = [],[],[]
    for train_ind, test_ind in kfold.split(x,y):
        ckp += 1
        X_train, X_test = x[train_ind], x[test_ind]
        Y_train, Y_test = y[train_ind], y[test_ind]
        
        if partial==True:
            estimator.partial_fit(X_train,Y_train)
        else:
            estimator.fit(X_train,Y_train)
            
        predictions = estimator.predict(X_test)
        acc = accuracy_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions, average='macro')
        mcc = matthews_corrcoef(Y_test, predictions)
        
        acc_lst += [acc]
        f1_lst += [f1]
        mcc_lst += [mcc]

        time = timestamp()

        if ckp_file is not None:
            with open(ckp_file, 'a') as f_output:
                f_output.write(f"{time}\n{estimator}, ckp: {ckp}\n acc: {acc}\n f1: {f1}\n mcc: {mcc}\n\n" )

    avg_acc = np.mean(acc_lst)
    avg_f1 = np.mean(f1_lst)
    avg_mcc = np.mean(mcc_lst)
    time = timestamp()
   
    with open(output, 'a') as f_output:
            f_output.write(f"{time}\nFinal Metrics for {estimator} \n Average acc: {avg_acc}\n Average F1: {avg_f1}\n Average MCC: {avg_mcc}\n\n")

    return estimator


def train_split_validation(estimator, X, y):
    '''
    Perform held-out test validation of ML model
    '''
    acc_lst, f1_lst, mcc_lst = [],[],[]
    
    random_states = [0, 333, 555, 777, 999]
    
    for state in random_states:
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                            train_size=0.82,
                                                            test_size=0.18, 
                                                            random_state=state, 
                                                            stratify=y)
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro')
        mcc = matthews_corrcoef(y_test, predictions)
        
        acc_lst += [acc]
        f1_lst += [f1]
        mcc_lst += [mcc]

        
        
    avg_acc = np.mean(acc_lst)
    avg_f1 = np.mean(f1_lst)
    avg_mcc = np.mean(mcc_lst)
    time = timestamp()
    
    results = {'acc': avg_acc, 'f1': avg_f1, 'mcc': avg_mcc }
    return f'time: {time} \n results: {results} \n'
    

def optimize_params(estimators, spaces, tuner, x, y, cv=5, split=1500, output=None):
    '''
    Performs hyperparameter optimization of ML models using grid search or Bayes search
    '''
    opt_hyperparams = []
    
    X_opt, Y_opt = x[:split], y[:split]
    
    
    search_grids = spaces['grids']
    search_spaces = spaces['bayes']
    
    keys = list(estimators.keys())
    
    if tuner == 'grid':
        for key, ind in zip(keys, range(len(keys))):
            model = estimators[key]
            model_params = search_grids[ind]
            
            search = GridSearchCV(
                estimator=model(),
                param_grid=model_params,
                cv=cv,
                verbose=False
            )
            
            search.fit(X_opt,Y_opt)
            opt_hp = search.best_params_
            opt_hyperparams += [opt_hp]
            
            time = timestamp()
            print(time)
            if output is not None:
                with open(output, 'a') as f_output:
                    f_output.write(f'{time}\n{model} {tuner} optimized hyperparamters: \n {opt_hp}\n\n')
                    
        return {tuner: dict(zip(keys, opt_hyperparams))}
    
    elif tuner == 'bayes':
        for key, ind in zip(keys,range(len(keys))):
            model = estimators[key]
            model_params = search_spaces[ind]
            
            search = BayesSearchCV(
                estimator=model(),
                search_spaces=model_params,
                cv=cv,
                n_iter=1,
                verbose=10
            )
            
            search.fit(X_opt,Y_opt)
            opt_hp = search.best_params_
            opt_hyperparams += [opt_hp]
            
            time = timestamp()
            print(time)
            if output is not None:
                with open(output, 'a') as f_output:
                    f_output.write(f'{time}\n{model} {tuner} optimized hyperparamters: \n{opt_hp}\n\n')
            
        return {tuner: dict(zip(keys, opt_hyperparams))}
    
    else:
        print('tuner must be grid or bayes')
        return None