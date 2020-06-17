import pandas as pd
import numpy as np

# Full power to apply funcs
from functools import partial
from multiprocessing import Pool, cpu_count

# Ploting
from matplotlib import pyplot as plt

# Sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.class_weight import compute_sample_weight

# PySurvival
from pysurvival.utils.metrics import concordance_index
from pysurvival.models.survival_forest import ConditionalSurvivalForestModel
from pysurvival.utils.display import compare_to_actual, integrated_brier_score, display_loss_values


class SurvivalLib():

    def __init__(self, df, target_col, del_features, best_features):
        print('Initializing...')
        self.df = df
        self.target = target_col
        self.features = [col for col in df.columns.tolist() 
                            if col not in del_features]
        self.event = 'EVENT'
        self.df[self.event] = df['OVERALL_SURVIVAL_STATUS'].apply(
                                lambda x: 1 if x == 'DECEASED' else 0)
        self.best_num_feats = 0
        self._num_feats_optimized = False        

        self.del_features = del_features

        self.best_features = best_features

        print('Done Initializing.')

    def reset_index(self):
        self.df = self.df.reset_index(drop=True)

    def clean_data(self, columns_to_drop,min_survival_months,
                    max_survival_months, verbose=True):
        '''
        Function to be run after initializing the class.
        Removes desired columns and filters time-axis.

        Parameters:
        ----------


        '''
        if verbose:
            print('Cleaning data...')
        self.columns_to_drop = columns_to_drop
        self.remove_missing_data(verbose)

        self.min_survival_months = min_survival_months
        self.max_survival_months = max_survival_months

        self.filter_data(verbose)

        self.reset_index()

        if verbose:
            print('Data cleaned.')

    def get_df(self):
        '''
        Function to return the current pandas DataFrame utilized.
        Useful to ploting.

        Returns:
        --------
        Current DataFrame.
        '''
        return self.df

    def remove_missing_data(self, verbose=True):

        na_cols = self.df.isna().any().reset_index()

        if verbose:
            print('Columns with NaN: \n', na_cols.loc[na_cols[0] == True]['index'])
            print('Shape before removing: ', self.df.shape)

        self.df = self.df.drop(self.columns_to_drop, axis=1).dropna()

        if verbose:
            print('Shape after removing non-important columns: ', self.df.shape)

    def filter_data(self, verbose=True):

        if verbose:
            print('Shape before filtering time-axis: ', self.df.shape)           
        filt = ((self.df[self.target] <= self.max_survival_months) & 
                (self.df[self.target] >= self.min_survival_months))
        self.df = self.df.loc[filt]
        if verbose:
            print('Shape after filtering time-axis: ', self.df.shape) 

    def set_best_num_feats(self, value):
        '''
        Sets the current best number of variables to train the model.

        Parameters
        ----------
        value: Number of features to use as the best one.

        '''
        self.best_num_feats = value

    def verify_best_num_feats(self):
        '''
        Function that verifies if the best number of features is
        being used, by checking the private variable _num_feats_optimized.
        If it's not optimized, the variable is initialized with 50.
        '''
        if self.best_num_feats == 0 | self._num_feats_optimized == False:
            print('Warning! Run function brute_force_num_features for best results! ' +
                    'Using default value of 50.')
            self.set_best_num_feats(50)
            return False
        else:
            return True

    def cv_survival(self, cv=10, params={}, return_model=False, 
                        num_trees=1000, verbose=True):

        self.verify_best_num_feats()

        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        models = []
        datasets = []
        df_cv = self.df.copy()
        dummies = list(filter(lambda x: x not in self.del_features, 
                              self.df.dtypes.loc[self.df.dtypes == 'object'].index))

        df_cv = pd.get_dummies(df_cv, columns=dummies).reset_index(drop=True)       

        features_new = [e for e in df_cv.columns.tolist() if e not in self.del_features]
        fold = 0
        for index_train, index_test in kf.split(df_cv):
            if verbose:
                print('Fold {}'.format(fold + 1))
                fold += 1
            data_train = df_cv.iloc[index_train].reset_index( drop = True )
            data_test  = df_cv.iloc[index_test].reset_index( drop = True )

            # Creating the X, T and E inputs
            X_train, X_test = data_train[features_new], data_test[features_new]
            T_train, T_test = data_train[self.target].values, data_test[self.target].values
            E_train, E_test = data_train[self.event].values, data_test[self.event].values

            X_train = X_train[self.best_features[:self.best_num_feats]]
            X_test = X_test[self.best_features[:self.best_num_feats]]

            weights = compute_sample_weight('balanced', E_train)
            weights = weights / df_cv.shape[0]
            weights[0] += 1 - sum(weights)
            
            model_forest = ConditionalSurvivalForestModel(num_trees=num_trees,)
            model_forest.fit(X_train, T_train, E_train, seed=42, weights=weights, **params)
            models.append(model_forest)
            datasets.append([X_test, T_test, E_test])
            scores.append(concordance_index(model_forest, X_test, T_test, E_test))

        if return_model:
            scores = np.array(scores)
            best_run = np.where(scores == max(scores))[0][0]
            best_model = models[best_run]
            best_datasets = datasets[best_run]
            return best_model, best_datasets, np.mean(scores)
        else:
            return np.mean(scores)


    def brute_force_num_features(self, min_feats=10, max_feats=150, verbose=True):
        '''
        Computes the best number of features to train the model.
        Uses brute force, so processing is slow.

        Parameters:
        -----------

        min_feats: Minimum number of features to start testing.
        max_feats: Maximum number of features to test.

        '''
        if verbose:
            print('Running number of features optimization...')
        best_res = 0
        best_feats = 0
        for num_feats in range(min_feats, max_feats, 10):
            c_index = self.cv_survival(df, cv=10, num_feats=num_feats, num_trees=1000)
            if c_index > best_res:
                best_res = c_index
                best_feats = num_feats
                if verbose: 
                    print('CV Score: {:.2f} Number of features: {}'.format(best_res, min_feats))        

        self.set_best_num_feats(best_feats)
        self._num_feats_optimized = True

        if verbose:
            print('Done optimizing.')




















