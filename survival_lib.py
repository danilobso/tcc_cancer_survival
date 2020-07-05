import pandas as pd
import numpy as np

import sys

# Full power to apply funcs
from functools import partial
from multiprocessing import Pool, cpu_count

# Ploting
from matplotlib import pyplot as plt

# Sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight

# PySurvival
from pysurvival.utils.metrics import concordance_index
from pysurvival.models.survival_forest import ConditionalSurvivalForestModel
from pysurvival.utils.display import compare_to_actual, integrated_brier_score, display_loss_values


class SurvivalLib():

    def __init__(self,
                 df,
                 target_col,
                 del_features,
                 feature_importance=None,
                 best_num_feats=0,
                 params={}):

        print('Initializing...')
        self.df = df
        self.target = target_col

        # Features will be processed after dumming columns
        self.features = []

        # Create event column
        self.event = 'EVENT'
        self.df[self.event] = self.df['OVERALL_SURVIVAL_STATUS'].apply(
                                lambda x: 1 if x == 'DECEASED' else 0)

        # Fix best number of features
        if best_num_feats == 0:
            self.best_num_feats = 50
            self._num_feats_optimized = False
        else:
            self.best_num_feats = best_num_feats
            self._num_feats_optimized = True

        self.del_features = del_features

        self.feature_importance = feature_importance
        if not isinstance(self.feature_importance, pd.Series):
            print('Please process feature importance for this dataset,')
            print('by calling SurvivalLib.process_feature_importance,')
            print('after cleaning data.')

        self.best_params = params

        self.test_size = 0.2

        self.seed = 42

        print('Done Initializing.')

    def data_split(self, verbose=True):
        """
        Function to separate data into train and test sets.
        All optimizations are done using train set, and
        all model evaluations are done using the test set.
        """

        dummies = list(filter(lambda x: x not in self.del_features,
                              self.df.dtypes.loc[self.df.dtypes == 'object'].index))

        self.df = pd.get_dummies(self.df, columns=dummies).reset_index(drop=True)

        self.features = [e for e in self.df.columns.tolist() if e not in self.del_features]

        df_train, df_test = train_test_split(self.df,
                                             test_size=self.test_size,
                                             shuffle=True,
                                             random_state=self.seed,
                                             stratify=self.df[self.event])
        self.df_train = df_train.reset_index(drop=True)
        self.df_test = df_test.reset_index(drop=True)

        if verbose:
            print('Data splited into train and test sets.')

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

        self.data_split(verbose)

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

    def get_X_test(self):
        return self.df_test[self.feature_importance[:self.best_num_feats]]

    def get_T_test(self):
        return self.df_test[self.target].values

    def get_E_test(self):
        return self.df_test[self.event].values

    def remove_missing_data(self, verbose=True):

        na_cols = self.df.isna().any().reset_index()

        if verbose:
            print('Columns with NaN: \n', na_cols.loc[na_cols[0] == True]['index'])
            print('Shape before removing: ', self.df.shape)

        self.df = self.df.drop(self.columns_to_drop, axis=1).dropna()

        # Fixing spaces on columns to be One Hot Encoded
        dummies = list(filter(lambda x: x not in self.del_features,
                              self.df.dtypes.loc[self.df.dtypes == 'object'].index))
        for col in dummies:
            self.df[col] = self.df[col].apply(lambda x: x.replace(' ', '_').replace(',','').upper())

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


    def check_params(self, params):
        if not params:
            return self.best_params
        else:
            return params

    def process_feature_importance(self, num_trees=4000, params={},
                                    balance_classes=True, verbose=True,):
        """
        Function to process the feature importance, in order to
        operate the library with a reduced number of features,
        a way to addess the curse of dimentionality.

        """

        params = self.check_params(params)

        print('Started processing feature importance. This may take a while.')

        df = self.df.copy()

        # Creating the X, T and E inputs
        X = df[self.features]
        T = df[self.target].values
        E = df[self.event].values

        weights = None

        if balance_classes:
            weights = compute_sample_weight('balanced', E)
            weights = weights / df.shape[0]
            weights[0] += 1 - sum(weights)

        model_forest = ConditionalSurvivalForestModel(num_trees=num_trees,)
        model_forest.fit(X, T, E, seed=self.seed, weights=weights, **params)

        self.set_feature_importance(model_forest.variable_importance_table['feature'])


    def set_feature_importance(self, df):
        self.feature_importance = df


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
        if (self.best_num_feats == 0) | (self._num_feats_optimized == False):
            print('Warning! Run function brute_force_num_features for best results! ' +
                    'Using default value of 50.')
            self.set_best_num_feats(50)
            return False
        else:
            return True

    def compute_event_weights(self, df, balance_classes=True):
        weights = None

        if balance_classes:
            weights = compute_sample_weight('balanced', df)
            weights = weights / df.shape[0]
            #weights[0] += 1 - sum(weights)

        return weights

    def cv_survival(self, cv=10, params={}, num_trees=1000,
                    balance_classes=True, verbose=True, ):

        self.verify_best_num_feats()

        # Check if the best hyperparameters were processed
        params = self.check_params(params)

        kf = KFold(n_splits=cv, shuffle=True, random_state=self.seed)

        scores = []
        models = []
        datasets = []

        df_cv = self.df_train.copy()

        for fold, (index_train, index_test) in enumerate(kf.split(df_cv), 1):
            if verbose:
                print('Fold {}'.format(fold))

            data_train = df_cv.iloc[index_train].reset_index( drop = True )
            data_test  = df_cv.iloc[index_test].reset_index( drop = True )

            # Creating the X, T and E inputs
            X_train, X_test = data_train[self.features], data_test[self.features]
            T_train, T_test = data_train[self.target].values, data_test[self.target].values
            E_train, E_test = data_train[self.event].values, data_test[self.event].values

            X_train = X_train[self.feature_importance[:self.best_num_feats]]
            X_test = X_test[self.feature_importance[:self.best_num_feats]]

            weights = self.compute_event_weights(E_train, balance_classes)

            # Creating model
            model_forest = ConditionalSurvivalForestModel(num_trees=num_trees,)
            model_forest.fit(X_train, T_train, E_train, seed=self.seed, weights=weights, **params)

            # Append score for post calculation average of folds
            scores.append(concordance_index(model_forest, X_test, T_test, E_test))

        # Refit model with all training data
        self.fit_model(num_trees=num_trees,
                       params=params,
                       balance_classes=balance_classes)

        scores = np.array(scores)
        self.cv_score = np.mean(scores)
        if verbose:
            print('CV Score: {:.3f}'.format(self.cv_score))

    def fit_model(self,
                  num_trees=3000,
                  params={},
                  balance_classes=True):
        """
        Function used to fit an usable model, with all
        trainig data, after doing parameters optimization.
        """

        features = self.get_X_test()
        target = self.get_T_test()
        event = self.get_E_test()

        weights = self.compute_event_weights(event,
                                             balance_classes)

        params = self.check_params(params)

        model_forest = ConditionalSurvivalForestModel(num_trees=num_trees,)

        model_forest.fit(features,
                          target,
                          event,
                          seed=self.seed,
                          weights=weights,
                          **params)

        self.model_forest = model_forest


    def get_cv_score(self):
        return self.cv_score

    def get_c_index(self):

        features = self.get_X_test()
        target = self.get_T_test()
        event = self.get_E_test()

        return concordance_index(self.model_forest, features,
                                    target, event)

    def optimize_hyperparams(self, verbose=True,):
        """
        Hyperparameter optimization. Default values are as follow:

        param_grid = {
            'min_node_size':[10],
            'max_depth':[5],
            'max_features':['sqrt']
            'minprop':[0.1],
            'alpha':[0.5],
            'sample_size_pct':[0.63],
            'importance_mode':['normalized_permutation']
        }
        """

        param_grid = {
            'min_node_size':[5, 7, 10],
            'max_depth':[4, 5, 6, 7],
            'max_features':['sqrt'],
            'minprop':[0.08, 0.1, 0.12],
            'alpha':[0.4, 0.5, 0.6, 0.7],
            'sample_size_pct':[0.63],
            'importance_mode':['normalized_permutation', 'impurity']
        }

        grid = ParameterGrid(param_grid)
        max_c_index = 0
        for i, params in enumerate(grid):
            self.cv_survival(cv=5, verbose=False, params=params)
            c_index = self.get_cv_score()
            if c_index > max_c_index:
                max_c_index = c_index
                self.best_params = params
                self.cv_score = max_c_index
                if verbose:
                    print('Run: {} CV Score: {:.3f}'.format(i+1, max_c_index))
        print(self.best_params)
        self.fit_model()

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

        self._num_feats_optimized = True
        best_res = 0
        best_feats = 0
        for num_feats in range(min_feats, max_feats, 10):
            self.set_best_num_feats(num_feats)
            self.cv_survival(cv=5, verbose=False)
            c_index = self.get_cv_score()
            if c_index > best_res:
                best_res = c_index
                best_feats = num_feats
                if verbose:
                    print('CV Score: {:.3f} Number of features: {}'.format(best_res, best_feats))

        self.set_best_num_feats(best_feats)

        if verbose:
            print('Done optimizing. Best number of features: {}'.format(best_feats))

        self.fit_model()


    def check_not_censored(self, sample):
        event = self.df_test[self.event].values
        if event[sample.name]:
            return True
        else:
            return False


    def predict_survival_all_times(self, sample, x_resolution=20):
        if self.check_not_censored(sample):
            preds = self.model_forest.predict_survival(sample)
            if isinstance(preds, np.ndarray):
                return preds[0]
            else:
                return None

    def predict_hazard_all_times(self, sample, x_resolution=20):
        if self.check_not_censored(sample):
            preds = self.model_forest.predict_hazard(sample)
            if isinstance(preds, np.ndarray):
                return preds[0]
            else:
                return None


    def predict_risk_all_samples(self, samples, x_resolution=20):
        preds = self.model_forest.predict_risk(samples)
        return preds

    def predict(self):
        features = self.get_X_test()
        res = features.apply(lambda x: self.predict_survival_all_times(x), axis=1)
        res = res[res.apply(lambda x: isinstance(x, np.ndarray))].reset_index(drop=True)
        self.preds = res
        return res

    def predict_survival_all_samples(self, sample):
        return self.model_forest.predict_survival(sample)

    def predict_risk(self):

        features = self.get_X_test()

        res = self.predict_risk_all_samples(features)

        self.risk_preds = features.apply(lambda x: self.predict_survival_all_times(x), axis=1)
        self.high_risk = np.where(res > np.median(res))[0]
        self.low_risk = np.where(res <= np.median(res))[0]
        self.plot_risk()
        return res

    def predict_hazard(self):
        features = self.get_X_test()

        res = features.apply(lambda x: self.predict_hazard_all_times(x), axis=1)
        res = res[res.apply(lambda x: isinstance(x, np.ndarray))].reset_index(drop=True)
        self.hazard_preds = res
        return res

    def get_preds(self):
        return self.preds

    def plot_risk(self):

        target = self.get_T_test()
        event = self.get_E_test()

        event_times = target[np.where(event == 1)[0]]
        f, ax = plt.subplots(figsize=(10,10))

        for pred in self.risk_preds.iloc[self.low_risk]:
            if isinstance(pred, np.ndarray):
                plt.plot(self.model_forest.times, pred, color='green', label='Baixo risco')
        for pred in self.risk_preds.iloc[self.high_risk]:
            if isinstance(pred, np.ndarray):
                plt.plot(self.model_forest.times, pred, color='red', label='Alto risco')

        kaplan = self.get_kaplan_curve()
        kaplan = kaplan.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_kaplan = scaler.fit_transform(kaplan)
        plt.plot(self.model_forest.times, scaled_kaplan, color='black')

        #print('Correct guesses: {:.2f}%'.format((count_right / i) * 100))
        #ax.legend()
        ax.set_ylabel('Probabilidade de sobrevivência')
        ax.set_xlabel('Tempo em meses após primeiro diagnóstico')
        ax.set_title('Comparação de curvas de sobrevivência para pacientes não censored')
        plt.show()


    def plot_prediction(self):

        features = self.get_X_test()
        target = self.get_T_test()
        event = self.get_E_test()

        event_times = target[np.where(event == 1)[0]]
        f, ax = plt.subplots(figsize=(10,10))
        i=0
        count_right = 0
        for pred in self.preds:
            if isinstance(pred, np.ndarray):
                if self.model_forest.times[np.argwhere(self.preds[0]<=0.5)[0][0]] <= event_times[i]:
                    plt.plot(self.model_forest.times, pred, color='green')
                    plt.scatter(event_times[i], 0.5, color='green')
                    count_right += 1
                else:
                    plt.scatter(event_times[i], 0.5, color='red')
                    plt.plot(self.model_forest.times, pred, color='red')
                i += 1
        print('Correct guesses: {:.2f}%'.format((count_right / i) * 100))
        ax.legend()
        ax.set_ylabel('Probabilidade de sobrevivência')
        ax.set_xlabel('Tempo em meses após primeiro diagnóstico')
        ax.set_title('Comparação de curvas de sobrevivência para pacientes não censored')
        plt.show()

    def plot_hazard(self):

        features = self.get_X_test()
        target = self.get_T_test()
        event = self.get_E_test()

        event_times = target[np.where(event == 1)[0]]
        f, ax = plt.subplots(figsize=(10,10))
        i=0
        count_right = 0
        for pred in self.hazard_preds:
            if isinstance(pred, np.ndarray):
                if self.model_forest.times[np.argwhere(self.preds[0]<=0.5)[0][0]] <= event_times[i]:
                    plt.plot(self.model_forest.times, pred, color='green')
                    #plt.scatter(event_times[i], 0.5, color='green')
                    count_right += 1
                else:
                    #plt.scatter(event_times[i], 0.5, color='red')
                    plt.plot(self.model_forest.times, pred, color='red')
                i += 1
        print('Correct guesses: {:.2f}%'.format((count_right / i) * 100))
        ax.set_ylabel('Hazard score')
        ax.set_xlabel('Tempo em meses após primeiro diagnóstico')
        ax.set_title('Comparação de curvas de sobrevivência para pacientes não censored')
        plt.show()

    def integrated_brier_score(self):
        integrated_brier_score(self.model_forest,
                               self.get_X_test(),
                               self.get_T_test(),
                               self.get_E_test(),
                               figure_size=(15,5))

    def compare_to_actual(self, is_at_risk=False):
        results = compare_to_actual(self.model_forest,
                                    self.get_X_test(),
                                    self.get_T_test(),
                                    self.get_E_test(),
                                    is_at_risk = is_at_risk,
                                    figure_size=(16, 6),
                                    metrics = ['rmse', 'mean', 'median'])


    def get_kaplan_curve(self):

        features = self.get_X_test()
        target = self.get_T_test()
        event = self.get_E_test()

        e_samples = list(features[features.apply(lambda sample: self.check_not_censored(sample), axis=1)].index)
        total_samples = len(e_samples)

        t_samples = []

        for time_bucket in self.model_forest.times:

            # Append current number of pacients
            t_samples.append(total_samples)

            # Verify if any event ocurred
            for sample in e_samples:
                if target[sample] <= time_bucket:

                    # Reduce total number of pacients
                    total_samples -= 1

                    # Remove this sample from list of samples
                    e_samples.remove(sample)

        return np.array(t_samples)

    def plot_kaplan(self):

        t_samples = self.get_kaplan_curve()
        f, ax = plt.subplots(figsize=(10,10))
        plt.plot(self.model_forest.times, t_samples)
        ax.set_ylabel('Number of current patients')
        ax.set_xlabel('Time from first diagnosis')
        ax.set_title('Kaplan survival function')
        plt.show()