import numpy as np
import pandas as pd
from os.path import join
from scipy.stats import rankdata
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

def _add_log(x): 
    return np.log(0.1+x)

class Data():
    C_FEATURES = [
        'HR', 'SysBP', 'MeanBP', 'DiaBP', 'Shock_Index',  # circulation status
        'Hb',  # volume depletion
        'BUN', 'Creatinine', 'output_4hourly',  # kidney perfusion
        'Arterial_pH', 'Arterial_BE', 'HCO3', 'Arterial_lactate',  # global perfusion
        'PaO2_FiO2',  # fluid tolerance
        'age', 'elixhauser', 'SOFA'
    ]
    C_TREATMENTS = ['input_4hourly', 'max_dose_vaso']
    C_GROUP = 'icustayid'
    C_OUTCOME = 'mortality_90d'

    C_SHIFT = ['gender', 'mechvent', 're_admission']
    C_LOG_SCALE = [
        'SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', 'INR',
        'input_total', 'input_4hourly', 'output_total', 'output_4hourly',
        'max_dose_vaso', 'input_4hourly_prev', 'max_dose_vaso_prev'
    ]
    C_SCALE = [
        'age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 
        'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride', 'Glucose', 
        'Magnesium', 'Calcium', 'Hb', 'WBC_count', 'Platelets_count', 'PTT', 
        'PT', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'HCO3', 'Arterial_lactate', 
        'SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2', 'cumulated_balance'
    ]
    C_SCALE += ['elixhauser']

    @staticmethod
    def get_scale_transform():
        return Pipeline(
            steps=[
                ('scale', StandardScaler())
            ]
        )

    @staticmethod
    def get_log_scale_transform():
        return Pipeline(
            steps=[
                ('logaritmize', FunctionTransformer(func=_add_log)),
                ('scale', StandardScaler())
            ]
        )

    @staticmethod
    def get_scale_columns(X):
        return sorted(list(set(X).intersection(Data.C_SCALE)))

    @staticmethod
    def get_log_scale_columns(X):
        return sorted(list(set(X).intersection(Data.C_LOG_SCALE)))

    @staticmethod
    def get_preprocessor():
        preprocessor = ColumnTransformer(
            transformers=[
                ('scale', Data.get_scale_transform(), Data.get_scale_columns),
                ('log_scale', Data.get_log_scale_transform(), Data.get_log_scale_columns)
            ],
            remainder='passthrough'
        )
        return preprocessor

    @staticmethod
    def discretize_treatments(treatments):
        '''
        Parameters
        ----------
        treatments : DataFrame of shape (n_samples, 2)
            Raw treatment data.

        Returns
        -------
        discrete_treatments : NumPy array of shape (n_samples, 2)
            Discrete treatments (values between 0 and 4).
        '''
        discrete_treatments = np.zeros(treatments.size)  # 0 is default (zero dose)
        is_nonzero = treatments > 0
        ranked_nonzero_treatments = rankdata(treatments[is_nonzero])/np.sum(is_nonzero)
        discrete_nonzero_treatments = np.digitize(ranked_nonzero_treatments, bins=[0., 0.25, 0.5, 0.75, 1.], right=True)
        discrete_treatments[is_nonzero] = discrete_nonzero_treatments
        return discrete_treatments

    @staticmethod
    def get_train_valid_test_data(data_path, aic_path, valid_size, test_size, use_aic_split=True, seed=None):
        sepsis_data = pd.read_csv(data_path)

        groups = sepsis_data[Data.C_GROUP]

        Y = sepsis_data[Data.C_TREATMENTS]
        Y_discrete = Y.apply(Data.discretize_treatments, raw=True)
        _, y = np.unique(Y_discrete, axis=0, return_inverse=True)
        n_classes = len(set(y))

        Yg = pd.concat([Y, groups], axis=1)
        previous_doses = Yg.groupby(by=Data.C_GROUP).apply(func=lambda df: df.shift(periods=1).fillna(0))
        previous_doses = previous_doses.drop(Data.C_GROUP, axis=1)
        rename = [(c, c+'_prev') for c in previous_doses.columns]
        previous_doses = previous_doses.rename(dict(rename), axis=1)

        X = sepsis_data[Data.C_FEATURES]
        X = pd.concat([X, previous_doses], axis=1)
        Xg = pd.concat([X, groups], axis=1)

        if use_aic_split:
            is_train = pd.read_csv(join(aic_path, 'is_train.csv'), header=None, names=['is_train']).squeeze('columns').astype('bool')
            Xg_train, y_train = Xg[is_train], y[is_train]
            Xg_test, y_test = Xg[~is_train], y[~is_train]
            groups_train = groups[is_train]
        else:
            assert test_size > 0
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            ii_train, ii_test = next(gss.split(X, y, groups))
            Xg_train, y_train = Xg.iloc[ii_train], y[ii_train]
            Xg_test, y_test = Xg.iloc[ii_test], y[ii_test]
            groups_train = groups.iloc[ii_train]

        if valid_size > 0:
            train_size = 1-test_size
            valid_size_ = valid_size/train_size
            gss = GroupShuffleSplit(n_splits=1, test_size=valid_size_, random_state=seed)
            ii_train, ii_valid = next(gss.split(Xg_train, y_train, groups_train))
            Xg_valid, y_valid = Xg_train.iloc[ii_valid], y_train[ii_valid]
            Xg_train, y_train = Xg_train.iloc[ii_train], y_train[ii_train]
        else:
            Xg_valid, y_valid = None, None

        states = pd.read_csv(join(aic_path, 'states.csv'), header=None, names=['state'])
        states -= 1  # To get states.min() == 0
        states_train = states.loc[Xg_train.index].to_numpy()
        states_valid = states.loc[Xg_valid.index].to_numpy() if valid_size > 0 else None
        states_test = states.loc[Xg_test.index].to_numpy()

        # Convert groups to sets since we only need unique IDs later
        groups_train = set(groups.loc[Xg_train.index])
        groups_valid = set(groups.loc[Xg_valid.index]) if valid_size > 0 else set()
        groups_test = set(groups.loc[Xg_test.index])

        outcomes = sepsis_data[Data.C_OUTCOME]
        rewards_train, rewards_valid, rewards_test = [], [], []
        for group, sequence in Xg.groupby(by=Xg.columns[-1]):
            outcomes_ = outcomes.loc[sequence.index].to_numpy()
            death = outcomes_[-1]  # 90-day mortality
            rewards_ = len(sequence.index) * [0]
            rewards_[-1] = -100 if death else 100
            if group in groups_train:
                rewards_train += rewards_
            elif group in groups_valid:
                rewards_valid += rewards_
            elif group in groups_test:
                rewards_test += rewards_
        rewards_train = np.array(rewards_train)
        rewards_valid =  np.array(rewards_valid)
        rewards_test = np.array(rewards_test)

        data_train = (Xg_train, y_train, states_train, rewards_train)
        data_valid = (Xg_valid, y_valid, states_valid, rewards_valid)
        data_test = (Xg_test, y_test, states_test, rewards_test)

        return data_train, data_valid, data_test
