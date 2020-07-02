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

    def __init__(self, df, target_col, del_features, feature_importance=None):
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

        self.feature_importance = feature_importance
        if not isinstance(self.feature_importance, pd.Series):
            print('Please process feature importance for this dataset,')
            print('by calling SurvivalLib.process_feature_importance,')
            print('after cleaning data.')

        self.test_size = 0.2

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

        # Fixing spaces on therapy columns
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


    def process_feature_importance(self, num_trees=4000, params={},
                                    balance_classes=True, verbose=True,):
        """
        Function to process the feature importance, in order to
        operate the library with a reduced number.

        """

        print('Started processing feature importance. This may take a while.')
        df = self.df.copy()
        dummies = list(filter(lambda x: x not in self.del_features,
                              self.df.dtypes.loc[self.df.dtypes == 'object'].index))

        df = pd.get_dummies(df, columns=dummies).reset_index(drop=True)
        N = df.shape[0]
        features_new = [e for e in df.columns.tolist() if e not in self.del_features]

        index_train, index_test = train_test_split(range(N),
                                           test_size = self.test_size,
                                           random_state=42)

        data_train = df.iloc[index_train].reset_index( drop = True )
        data_test  = df.iloc[index_test].reset_index( drop = True )

        # Creating the X, T and E inputs
        X_train, X_test = data_train[features_new], data_test[features_new]
        T_train, T_test = data_train[self.target].values, data_test[self.target].values
        E_train, E_test = data_train[self.event].values, data_test[self.event].values

        weights = None

        if balance_classes:
            weights = compute_sample_weight('balanced', E_train)
            weights = weights / df.shape[0]
            weights[0] += 1 - sum(weights)

        model_forest = ConditionalSurvivalForestModel(num_trees=num_trees,)
        model_forest.fit(X_train, T_train, E_train, seed=42, weights=weights, **params)

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
        if self.best_num_feats == 0 | self._num_feats_optimized == False:
            print('Warning! Run function brute_force_num_features for best results! ' +
                    'Using default value of 50.')
            self.set_best_num_feats(50)
            return False
        else:
            return True

    def cv_survival(self, cv=10, params={}, num_trees=1000,
                    balance_classes=True, verbose=True, ):

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

            X_train = X_train[self.feature_importance[:self.best_num_feats]]
            X_test = X_test[self.feature_importance[:self.best_num_feats]]

            weights = None

            if balance_classes:
                weights = compute_sample_weight('balanced', E_train)
                weights = weights / df_cv.shape[0]
                weights[0] += 1 - sum(weights)

            model_forest = ConditionalSurvivalForestModel(num_trees=num_trees,)
            model_forest.fit(X_train, T_train, E_train, seed=42, weights=weights, **params)
            models.append(model_forest)
            datasets.append([X_test, T_test, E_test])
            scores.append(concordance_index(model_forest, X_test, T_test, E_test))

        scores = np.array(scores)
        best_run = np.where(scores == max(scores))[0][0]
        best_model = models[best_run]
        best_datasets = datasets[best_run]
        self.model_forest = best_model
        self.X_test, self.T_test, self.E_test = best_datasets
        print('CV Score: {:.3f}'.format(np.mean(scores)))


    def get_c_index(self):
        return concordance_index(self.model_forest, self.X_test,
                                    self.T_test, self.E_test)

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
            self.set_best_num_feats(num_feats)
            c_index = self.cv_survival(verbose=False)
            if c_index > best_res:
                best_res = c_index
                best_feats = num_feats
                if verbose:
                    print('CV Score: {:.3f} Number of features: {}'.format(best_res, best_feats))

        self.set_best_num_feats(best_feats)
        self._num_feats_optimized = True

        if verbose:
            print('Done optimizing.')


    def check_not_censored(self, sample):
        if self.E_test[sample.name]:
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
        res = self.X_test.apply(lambda x: self.predict_survival_all_times(x), axis=1)
        res = res[res.apply(lambda x: isinstance(x, np.ndarray))].reset_index(drop=True)
        self.preds = res
        return res

    def predict_survival_all_samples(self, sample):
        return self.model_forest.predict_survival(sample)

    def predict_risk(self):
        res = self.predict_risk_all_samples(self.X_test)
        self.risk_preds = self.X_test.apply(lambda x: self.predict_survival_all_times(x), axis=1)
        self.high_risk = np.where(res > np.mean(res))[0]
        self.low_risk = np.where(res <= np.mean(res))[0]
        self.plot_risk()
        return res

    def predict_hazard(self):
        res = self.X_test.apply(lambda x: self.predict_hazard_all_times(x), axis=1)
        res = res[res.apply(lambda x: isinstance(x, np.ndarray))].reset_index(drop=True)
        self.hazard_preds = res
        return res

    def get_preds(self):
        return self.preds

    def plot_risk(self):
        event_times = self.T_test[np.where(self.E_test == 1)[0]]
        f, ax = plt.subplots(figsize=(10,10))

        for pred in self.risk_preds.iloc[self.low_risk]:
            if isinstance(pred, np.ndarray):
                plt.plot(self.model_forest.times, pred, color='green', label='Baixo risco')
        for pred in self.risk_preds.iloc[self.high_risk]:
            if isinstance(pred, np.ndarray):
                plt.plot(self.model_forest.times, pred, color='red', label='Alto risco')

        #print('Correct guesses: {:.2f}%'.format((count_right / i) * 100))
        #ax.legend()
        ax.set_ylabel('Probabilidade de sobrevivência')
        ax.set_xlabel('Tempo em meses após primeiro diagnóstico')
        ax.set_title('Comparação de curvas de sobrevivência para pacientes não censored')
        plt.show()


    def plot_prediction(self):
        event_times = self.T_test[np.where(self.E_test == 1)[0]]
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
        event_times = self.T_test[np.where(self.E_test == 1)[0]]
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







