import numpy as np
import pandas as pd
import optuna
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import traceback

# Custom scoring function for Gini coefficient
def gini_score(y_true, y_prob):
    return 2 * roc_auc_score(y_true, y_prob) - 1

gini_scorer = make_scorer(gini_score, needs_proba=True)

class ModelOptimizer:
    def __init__(self, models, X_train, y_train, n_trials=5, early_stopping_rounds=10):
        self.models = models
        self.X_train = X_train
        self.y_train = y_train
        self.n_trials = n_trials
        self.early_stopping_rounds = early_stopping_rounds
        self.best_params = {}
        self.best_models = {}

    def objective(self, trial, model_name):
        if model_name == 'naive_bayes':
            nb_type = trial.suggest_categorical('nb_type', ['gaussian', 'bernoulli'])
            if nb_type == 'gaussian':
                model = GaussianNB()

            elif nb_type == 'bernoulli':
                alpha = trial.suggest_loguniform('alpha', 1e-3, 1e0)
                binarize = trial.suggest_loguniform('binarize', 1e-3, 1e0)
                model = BernoulliNB(alpha=alpha, binarize=binarize)
                
        elif model_name == 'xgboost':
            n_estimators = trial.suggest_int('n_estimators', 20, 200)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
            model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric='logloss')
            
        elif model_name == 'random_forest':
            n_estimators = trial.suggest_int('n_estimators', 20, 200)
            max_depth = trial.suggest_int('max_depth', 5, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            
        elif model_name == 'logistic_regression':
            C = trial.suggest_loguniform('C', 1e-3, 1e2)
            solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga', 'sag'])
            penalty = None
            l1_ratio = None

            if solver == 'liblinear':
                penalty = trial.suggest_categorical('penalty_liblinear', ['l1', 'l2'])
            elif solver == 'saga':
                penalty = trial.suggest_categorical('penalty_saga', ['l1', 'l2', 'elasticnet', None])
                if penalty == 'elasticnet':
                    l1_ratio = trial.suggest_float('l1_ratio', 0, 1)

            model = LogisticRegression(C=C, penalty=penalty, solver=solver, l1_ratio=l1_ratio, max_iter=10000)

        skf = StratifiedKFold(n_splits=3)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring=gini_scorer)
        return scores.mean()

    def optimize_model(self, model_name):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, model_name), n_trials=self.n_trials, 
                       callbacks=[self.early_stopping_callback])
        return study.best_params

    def early_stopping_callback(self, study, trial):
        if len(study.trials) > self.early_stopping_rounds:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            completed_trials = completed_trials[-self.early_stopping_rounds:]
            values = [t.value for t in completed_trials]

            if len(values) < self.early_stopping_rounds:
                return

            if all(x >= values[0] for x in values):
                print("Early stopping triggered.")
                study.stop()

    def train_best_model(self, model_name, best_params):
        # Filter out extra keys for Logistic Regression
        if model_name == 'logistic_regression':
            best_params = {k: v for k, v in best_params.items() if k in ['C', 'solver', 'penalty', 'l1_ratio']}

        if model_name == 'naive_bayes':
            nb_type = best_params.pop('nb_type')
            if nb_type == 'gaussian':
                model = GaussianNB(**best_params)
            elif nb_type == 'multinomial':
                model = MultinomialNB(**best_params)
            elif nb_type == 'bernoulli':
                model = BernoulliNB(**best_params)
        elif model_name == 'xgboost':
            model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        elif model_name == 'random_forest':
            model = RandomForestClassifier(**best_params)
        elif model_name == 'logistic_regression':
            model = LogisticRegression(**best_params, max_iter=10000)
        model.fit(self.X_train, self.y_train)
        return model

    def optimize_and_train_model(self, model_name):
        try:
            print(f"Optimizing {model_name}...")
            best_params = self.optimize_model(model_name)
            print(f"Best parameters for {model_name}: {best_params}")
            best_model = self.train_best_model(model_name, best_params)
            return model_name, best_params, best_model
        except Exception as e:
            print(f"Error in optimizing and training {model_name}: {e}")
            traceback.print_exc()
            return model_name, None, None

    def run(self):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.optimize_and_train_model, model) for model in self.models]
            for future in futures:
                model_name, params, model = future.result()
                if params is not None and model is not None:
                    self.best_params[model_name] = params
                    self.best_models[model_name] = model

        return self.best_params, self.best_models

# Example usage
# X_train, y_train = your training data here
# models = ['naive_bayes', 'xgboost', 'random_forest', 'logistic_regression']
# optimizer = ModelOptimizer(models, X_train, y_train, n_trials=50, early_stopping_rounds=10)
# best_params, best_models = optimizer.run()
