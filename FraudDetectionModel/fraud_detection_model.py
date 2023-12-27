
from datetime import datetime
from pathlib import Path
from statistics import mode
from typing import List, Tuple
import json
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier, Pool
from numpy import where, arange, argmax, rint
from pandas import DataFrame
from sklearn.metrics import (PrecisionRecallDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import MinMaxScaler
import optuna


ITERATION = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
p = Path(f"results/{ITERATION}")
p.mkdir(parents=True, exist_ok=True)
scores_path = Path(f"results/scores_params.csv")
scores_path.touch(exist_ok=True)


class FraudDetection():
    """Fraud detection model of transactional data"""
    def __init__(self) -> None:
        base_path = Path(__file__).parent
        DATA_DIR = 'data'
        self.LABELS = base_path / DATA_DIR / 'labels.csv'
        self.TRANSAC = base_path / DATA_DIR / 'transactions.csv'

    def run(self):
        """Execute training"""
        data = self.data_loader()
        data = self.define_label(data)
        train, validation, holdout = self.splitter_ts(data)
        
        train = self.feature_engineering(train)
        validation = self.feature_engineering(validation)
        holdout = self.feature_engineering(holdout)
        num_features, cat_features = self.get_features(train)

        self.display_data_set_dimensions(data, train, validation, holdout)
        
        tuned_model, data_sets = self.fit_model(train, validation, holdout, num_features, cat_features)
        self.display_results(tuned_model, data_sets)
        
    def data_loader(self) -> DataFrame:
        """Load data, combine labels with transaction data, fill in NaNs."""
        labels = pd.read_csv(self.LABELS)
        transac = pd.read_csv(self.TRANSAC)

        labels['reportedTime'] = pd.to_datetime(labels['reportedTime'])
        transac['transactionTime'] = pd.to_datetime(transac['transactionTime'])

        assert len(transac['eventId'].unique()) == len(transac['eventId'])
        assert len(labels['eventId'].unique()) == len(labels['eventId'])

        data = pd.merge(transac, labels, how='left', on='eventId')
        return data

    def define_label(self, data: DataFrame) -> DataFrame:
        data['reportedTime'] = data['reportedTime'].fillna(-1)
        data['target'] = where(data['reportedTime'] == -1, 0, 1)
        data = data.drop(['eventId'], axis=1)

        data = data.fillna(-1)
        data = data.sort_index()
        data = data.reset_index(drop=True)
        print(f'Features available ({len([f for f in data.columns])}): {[f for f in data.columns]}\n')
        return data

    def splitter_ts(self, data: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        data['transactionTime'] = pd.to_datetime(data['transactionTime'])
        data = data.sort_values(by='transactionTime', ascending=True)
        data = data.set_index('transactionTime')
        train = data[data.index < '2017-10-31T23:59:59']
        validation = data[(data.index >= '2017-11-01T00:00:00') & (data.index < '2017-12-31T23:59:59')]
        holdout = data[data.index > '2017-12-31T23:59:59']
        train = train.reset_index()
        validation = validation.reset_index()
        holdout = holdout.reset_index()
        return train, validation, holdout

    def feature_engineering(self, data: DataFrame) -> DataFrame:
        """Create new features from raw data"""
        data = self.get_date_features(data)

        _mode = data.groupby('accountNumber').apply(lambda group: mode(group['posEntryMode'])).reset_index()
        _mode = _mode.rename({0: 'mode_entryMode'}, axis=1)
        data = pd.merge(data, _mode, on='accountNumber')

        _avg = data.groupby('accountNumber').apply(lambda group: group['transactionAmount'].mean()).reset_index()
        _avg = _avg.rename({0: 'avg_transac'}, axis=1)
        data = pd.merge(data, _avg, on='accountNumber')

        _min = data.groupby('accountNumber').apply(lambda group: group['transactionAmount'].min()).reset_index()
        _min = _min.rename({0: 'min_transac'}, axis=1)
        data = pd.merge(data, _min, on='accountNumber')

        _max = data.groupby('accountNumber').apply(lambda group: group['transactionAmount'].max()).reset_index()
        _max = _max.rename({0: 'max_transac'}, axis=1)
        data = pd.merge(data, _max, on='accountNumber')

        _transac_daily = data.groupby(['accountNumber', 'transaction_year', 'transaction_month', 'transaction_day']).apply(lambda x: len(x)).reset_index()
        _transac_daily = _transac_daily.rename({0: 'transac_daily'}, axis=1)
        data = pd.merge(data, _transac_daily, on=['accountNumber', 'transaction_year', 'transaction_month', 'transaction_day'])
        
        def time_between(group):
            return (group['transactionTime'] - group['transactionTime'].shift()).dt.total_seconds().fillna(0)

        _time_between = data.groupby('accountNumber').apply(time_between)
        _time_between = _time_between.reset_index().rename({'transactionTime': 'time_between'}, axis=1)
        data['time_between'] = _time_between['time_between']

        def merchant_country_changes(group):
            _mc = group.loc[group.index[0], 'merchantCountry']
            return group.loc[group.index[0]:, 'merchantCountry'] == _mc

        _mcc = data.groupby('accountNumber').apply(merchant_country_changes)
        data['merchantCountryChanges'] = _mcc.reset_index().rename({'merchantCountry': 'merchantCountryChanges'}, axis=1)['merchantCountryChanges']
        data['merchantCountryChanges'] = data['merchantCountryChanges'] * 1.0
        return data

    def get_features(self, data: DataFrame) -> Tuple[List, List]:
        """Get list of features available"""
        categorical_features = data.dtypes[data.dtypes != 'int64'][data.dtypes != 'float64'][data.dtypes != 'int32']
        categorical_features = [c for c in data.columns if c in categorical_features]
        numeric_features = [c for c in data.columns if c not in categorical_features]
        categorical_features.remove('transactionTime')
        categorical_features.remove('reportedTime')
        categorical_features.append('merchantCountry')
        categorical_features.append('mcc')
        categorical_features.append('posEntryMode')
        numeric_features.remove('target')
        numeric_features.remove('transaction_year')
        numeric_features.remove('merchantCountry')
        numeric_features.remove('mcc')
        numeric_features.remove('posEntryMode')
        
        categorical_features.remove('accountNumber')
        
        return numeric_features, categorical_features

    def get_date_features(self, data: DataFrame) -> DataFrame:
        """Extract date features from the transaction date."""
        data['transaction_year'] = data['transactionTime'].dt.year
        data['transaction_month'] = data['transactionTime'].dt.month
        data['transaction_day'] = data['transactionTime'].dt.day
        data['transaction_hour'] = data['transactionTime'].dt.hour
        data['transaction_weekday'] = data['transactionTime'].dt.dayofweek
        return data

    def display_data_set_dimensions(self, data, train, validation, holdout):
        """Give info about the data sets."""
        print(f'### Data set shapes ###\nTrain dimensions: {train.shape}\nValidation dimensions: {validation.shape}')
        print(f'Holdout dimensions: {holdout.shape}')
        assert train.shape[0] + validation.shape[0] + holdout.shape[0] == data.shape[0]

        __df = DataFrame({
            'train': [train['target'].sum(), len(train)], 
            'validation': [validation['target'].sum(), len(validation)], 
            'holdout': [holdout['target'].sum(), len(holdout)]
            }, index=['reportedTransactions', 'UnknownTransactions']
        )

        print(f"### Reported Transactions in each data set\n {__df}")

    def fit_model(
        self,
        train: DataFrame,
        validation: DataFrame,
        holdout: DataFrame,
        numeric_features: List,
        cat_features: List,
    ):
        """Train and optimize model """
        train = train.reset_index(drop=False)
        X = train[numeric_features + cat_features]
        y = train['target']

        X_val = validation[numeric_features + cat_features]
        y_val = validation['target']

        X_holdout = holdout[numeric_features + cat_features]
        y_holdout = holdout['target']

        data_sets = {}
        data_sets["X_train"] = X
        data_sets["y_train"] = y
        data_sets["X_val"] = X_val
        data_sets["y_val"] = y_val
        data_sets["X_holdout"] = X_holdout
        data_sets["y_holdout"] = y_holdout

        train_pool = Pool(X, y, cat_features=cat_features)
        eval_pool = Pool(X_val, y_val, cat_features=cat_features)

        data_sets["train_pool"] = train_pool
        data_sets["eval_pool"] = eval_pool
        
        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 150, 500),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "border_count": trial.suggest_int("border_count", 1, 255),
                "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 2, 30),
                "random_strength": trial.suggest_float("random_strength", 1e-9, 10),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
                "scale_pos_weight": trial.suggest_int("scale_pos_weight", 100, 300),
                "used_ram_limit": "5gb"
            }

            tuned_model = CatBoostClassifier(**params, verbose=100)
            tuned_model.fit(X, y, eval_set=eval_pool, early_stopping_rounds=50, cat_features=cat_features)

            preds = tuned_model.predict_proba(data_sets["X_val"])[:, 1]
            pred_labels = rint(preds)
            f1_scores = f1_score(data_sets['y_val'], pred_labels)
            return f1_scores

        needs_tuning = False
        if needs_tuning:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=10, timeout=600)
            print("Number of finished trials: {}".format(len(study.trials)))
            print("Best trial:")
            trial = study.best_trial
            print("  Value: {}".format(trial.value))
            print("  Params: ")
            for key, value in trial.params.items():
                print("{}: {}".format(key, value))

            params_df = pd.DataFrame({'iteration': ITERATION, 'trial_value': trial.value, 'params': [study.best_params]}, index=[0])
            
            scores_file = pd.read_csv(scores_path)
            scores_file = pd.concat([scores_file, params_df], ignore_index=True)
            scores_file.to_csv(scores_path, index=False)

            tuned_model = CatBoostClassifier(**study.best_params)
        else:
            scores_file = pd.read_csv(scores_path)
            _max = max(scores_file['trial_value'])
            params = eval(scores_file.loc[scores_file['trial_value'] == _max, 'params'][0])
            
            tuned_model = CatBoostClassifier(**params, verbose=100)
            
        tuned_model.fit(X, y, cat_features=cat_features)
        return tuned_model, data_sets

    def display_results(self, tuned_model, data_sets):
        """Plot out results and save to folder."""
        f_imp = DataFrame({'feature_importance': tuned_model.get_feature_importance(data_sets["train_pool"]),
              'feature_names': data_sets["X_train"].columns}).sort_values(by=['feature_importance'], 
                                                           ascending=False)
        _, ax = plt.subplots()
        f_imp.sort_values(by=['feature_importance'], ascending=True).plot.barh(x='feature_names', y='feature_importance')
        plt.savefig(f'results/{ITERATION}/feature_imp.png', bbox_inches='tight')

        pred_scores_train = DataFrame(tuned_model.predict_proba(data_sets["X_train"])[:, 1], columns=['probability_scores'])
        pred_scores_val = DataFrame(tuned_model.predict_proba(data_sets["X_val"])[:, 1], columns=['probability_scores'])

        _, ax = plt.subplots()
        PrecisionRecallDisplay.from_predictions(data_sets["y_train"], pred_scores_train['probability_scores'], ax=ax)
        PrecisionRecallDisplay.from_predictions(data_sets["y_val"], pred_scores_val['probability_scores'], ax=ax)
        plt.savefig(f'results/{ITERATION}/train_val.png', bbox_inches='tight')

        holdout_results = DataFrame(tuned_model.predict_proba(data_sets["X_holdout"])[:, 1], columns=['probability_scores'])
        holdout_results['true_value'] = data_sets["y_holdout"]
        holdout_results['transactionAmount'] = data_sets["X_holdout"].reset_index()['transactionAmount']
        holdout_results['total_fraud_transacted'] = holdout_results.loc[(holdout_results['true_value'] == 1), 'transactionAmount'].sum()
        total_fraud_transacted = holdout_results.loc[(holdout_results['true_value'] == 1), 'transactionAmount'].sum()

        def to_labels(pos_probs, threshold):
            return (pos_probs >= threshold).astype('int')

        _step = 0.02
        thresholds = arange(0, 1, _step)
        scores = [f1_score(holdout_results['true_value'], to_labels(holdout_results['probability_scores'], t)) for t in thresholds]
        
        # get best threshold
        ix = argmax(scores)
        print('Threshold=%.2f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

        saved_list = []
        for i in arange(0, 1, _step):
            holdout_results[f'prediction_{100*i}'] = holdout_results['probability_scores'] > i
            saved_list.append(holdout_results.loc[(holdout_results[f'prediction_{100*i}'] == 1) & (holdout_results['true_value'] == 1), 'transactionAmount'].sum())
            # holdout_results[f'saved_{i}'] = holdout_results.loc[(holdout_results[f'prediction_{i}'] == 1) & (holdout_results['true_value'] == 1), 'transactionAmount'].sum()

        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.scatter(arange(0, 1, _step), scores, c='b')
        ax2.scatter(arange(0, 1, _step), saved_list, c='r')
        ax2.plot(arange(0, 1, _step), saved_list, c='r')
        plt.title('F1 score (left) and money saved (right) at various thresholds (x-axis)')
        plt.savefig(f'results/{ITERATION}/f1_savings.png', bbox_inches='tight')

        _df = DataFrame(MinMaxScaler().fit_transform(DataFrame(data={"scores": scores, "savings": saved_list})), columns=["scores", "savings"])
        _df["added"] = _df["scores"] + _df["savings"]
        _argmax  = 2*argmax(_df["added"])

        amount_saved = holdout_results.loc[(holdout_results[f'prediction_{_argmax}.0'] == 1) & (holdout_results['true_value'] == 1), 'transactionAmount'].sum()
        print(confusion_matrix(holdout_results['true_value'], holdout_results[f'prediction_{_argmax}.0']))
        print(classification_report(holdout_results['true_value'], holdout_results[f'prediction_{_argmax}.0']))
        print(f'Amount saved: {amount_saved}')
        print(f'Total fraud transacted: {total_fraud_transacted}')
        print(f'value captured: {amount_saved/total_fraud_transacted}')
        


if __name__ == '__main__':
    FraudDetection().run()
