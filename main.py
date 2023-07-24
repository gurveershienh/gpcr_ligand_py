import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from custom_funcs import computeFP, train_split_validation, optimize_params
from configuration import search_params, learners
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def main():

    #load and featurize molecules
    data = pd.read_csv('data/basic_descriptors.csv')
    df = computeFP(data['canonical_smiles'], data['gpcr_binding_encoded'])
    
    Y = df['labels'].to_numpy()
    X = df.drop(['labels'], axis=1).to_numpy()

    X, Y = shuffle(X,Y)

    output = 'ecfp6_training.out'
    
    keys = list(learners.keys())
    
    ##optimize_params returns a nested dictionary with optimized hyperparameters
    hyperparameters = optimize_params(
        estimators=learners,
        spaces=search_params,
        tuner='grid',
        x=X,
        y=Y,
        cv=5,
        output=output
    )
    
    ##train and validate each model
    for key in keys:
        
        learner = learners[key]
        opt_params = hyperparameters['grid'][key]
        opt_learner = learner(**opt_params)
        
        model_metrics = train_split_validation(
            estimator=opt_learner,
            x=X,
            y=Y
        )
        
        ##record results
        with open(output, 'a') as f:
            f.write(f'{key}, {model_metrics}')
    return
            

        
        
                
if __name__ == "__main__":
    main()