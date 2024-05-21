import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, make_scorer
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
from collections import defaultdict

class ModelEvaluator:
    def __init__(self, Xtrain, Ytrain, Xtest, Ytest, best_models):
        self.Xtrain = pd.DataFrame(Xtrain).drop_duplicates()
        self.Ytrain = pd.Series(Ytrain).loc[self.Xtrain.index]
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.best_models = best_models
        self.train_results_df = pd.DataFrame()
        self.test_results_df = pd.DataFrame()
        self._initialize_results()

    def _initialize_results(self):
        self.train_results_df['Y_Train'] = self.Ytrain.reset_index(drop=True)
        self.test_results_df['Y_Test'] = pd.Series(self.Ytest).reset_index(drop=True)

    def evaluate_models(self):
        for model_name, model in self.best_models.items():
            y_train_pred = model.predict(self.Xtrain)
            y_train_pred_proba = model.predict_proba(self.Xtrain)[:, 1]
            
            y_test_pred = model.predict(self.Xtest)
            y_test_pred_proba = model.predict_proba(self.Xtest)[:, 1]
            
            self.train_results_df[model_name + '_Train_Pred'] = y_train_pred
            self.train_results_df[model_name + '_Train_Proba'] = y_train_pred_proba

            self.test_results_df[model_name + '_Test_Pred'] = y_test_pred
            self.test_results_df[model_name + '_Test_Proba'] = y_test_pred_proba

    def plot_roc_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for i, model_name in enumerate(self.best_models.keys()):
            fpr_train, tpr_train, _ = roc_curve(self.train_results_df['Y_Train'], self.train_results_df[model_name + '_Train_Proba'])
            fpr_test, tpr_test, _ = roc_curve(self.test_results_df['Y_Test'], self.test_results_df[model_name + '_Test_Proba'])
            
            train_auc = roc_auc_score(self.train_results_df['Y_Train'], self.train_results_df[model_name + '_Train_Proba'])
            train_gini = 2 * train_auc - 1
            test_auc = roc_auc_score(self.test_results_df['Y_Test'], self.test_results_df[model_name + '_Test_Proba'])
            test_gini = 2 * test_auc - 1
            
            ax = axes[i]
            ax.plot(fpr_train, tpr_train, label=f'Train (AUC = {train_auc:.2f}, Gini = {train_gini:.2f})')
            ax.plot(fpr_test, tpr_test, label=f'Test (AUC = {test_auc:.2f}, Gini = {test_gini:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve for {model_name}')
            ax.legend(loc='best')

        plt.tight_layout()
        plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, make_scorer
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
from collections import defaultdict

class FeatureImportance:
    def __init__(self, train_upsampled, best_models, forforsta):
        self.train_upsampled = train_upsampled
        self.best_models = best_models
        self.forforsta = forforsta
        self.feature_names = train_upsampled.drop(columns='Ever90').columns
        self.gini_scorer = make_scorer(self.gini_score, needs_proba=True)
        self.normalized_importances = {}
        self.cumulative_importance = {}

    @staticmethod
    def gini_score(y_true, y_prob):
        return 2 * roc_auc_score(y_true, y_prob) - 1

    @staticmethod
    def normalize_importances(importances):
        total_importance = sum(abs(value) for value in importances.values())
        return {k: abs(v) / total_importance for k, v in importances.items()}

    def calculate_importances(self):
        # Random Forest
        rfr_tune = self.best_models['random_forest']
        feature_importances = rfr_tune.feature_importances_
        importance_dict = dict(zip(self.feature_names, feature_importances))
        self.normalized_importances['RFS'] = self.normalize_importances(importance_dict)

        # Logistic Regression
        lr_tuned = self.best_models['logistic_regression']
        coefficients = lr_tuned.coef_[0]
        importance_dict = dict(zip(self.feature_names, abs(coefficients)))
        self.normalized_importances['LG'] = self.normalize_importances(importance_dict)

        # XGBoost
        xgb_model = self.best_models['xgboost']
        feature_importances = xgb_model.feature_importances_
        importance_dict = dict(zip(self.feature_names, feature_importances))
        self.normalized_importances['XGB'] = self.normalize_importances(importance_dict)

        # Naive Bayes
        nb_model = self.best_models['naive_bayes']
        train_features = self.train_upsampled.drop(columns='Ever90')
        train_target = self.train_upsampled['Ever90']
        result = permutation_importance(nb_model, train_features, train_target, scoring=self.gini_scorer, n_repeats=10, random_state=42)
        importance_dict = dict(zip(self.feature_names, result.importances_mean))
        self.normalized_importances['Naive Bayes'] = self.normalize_importances(importance_dict)

    def calculate_cumulative_importance(self):
        combined_dict = {
            'RFS': self.normalized_importances['RFS'],
            'LG': self.normalized_importances['LG'],
            'XGB': self.normalized_importances['XGB'],
            'Naive Bayes': self.normalized_importances['Naive Bayes']
        }

        self.cumulative_importance = {feature: sum(abs(importances.get(feature, 0)) for importances in combined_dict.values()) for feature in self.feature_names}

    def find_highly_correlated_features(self):
        correlation_matrix = self.forforsta[self.forforsta.columns].corr()
        highly_correlated_pairs = []

        for col in correlation_matrix.columns:
            for idx in correlation_matrix.index:
                if 0.85 < correlation_matrix[col][idx] < 1:
                    highly_correlated_pairs.append((col, idx))

        groups = defaultdict(set)

        for col, idx in highly_correlated_pairs:
            found = False
            for group in groups.values():
                if col in group or idx in group:
                    group.add(col)
                    group.add(idx)
                    found = True
                    break
            if not found:
                groups[len(groups)] = {col, idx}

        highly_correlated_features = [list(group) for group in groups.values()]

        features_to_keep = []
        features_to_discard = []

        for group in highly_correlated_features:
            if not group:
                continue
            sorted_group = sorted(group, key=lambda x: self.cumulative_importance.get(x, 0), reverse=True)
            features_to_keep.append(sorted_group[0])
            features_to_discard.extend(sorted_group[1:])

        return features_to_keep, features_to_discard

    def plot_importances(self):
        combined_dict = {
            'RFS': self.normalized_importances['RFS'],
            'LG': self.normalized_importances['LG'],
            'XGB': self.normalized_importances['XGB'],
            'Naive Bayes': self.normalized_importances['Naive Bayes']
        }

        cumulative_importance = {feature: sum(abs(importances.get(feature, 0)) for importances in combined_dict.values()) for feature in self.feature_names}
        sorted_features = sorted(cumulative_importance, key=cumulative_importance.get, reverse=True)

        fig = go.Figure()

        for method, importances in combined_dict.items():
            values = [importances.get(feature, 0) for feature in sorted_features]
            fig.add_trace(go.Bar(
                x=sorted_features,
                y=values,
                name=method
            ))

        fig.update_layout(
            title="Normalized Feature Importances by Algorithm",
            barmode='stack',
            xaxis_title="Features",
            yaxis_title="Importance",
            xaxis={'categoryorder': 'total descending', 'tickangle': 50},
            width=1800,
            height=800
        )

        fig.show()

        return sorted_features 

# # Usage
# # Assuming best_models, train_upsampled, and forforsta are predefined
# feature_importance = FeatureImportance(train_upsampled, best_models, forforsta)
# feature_importance.calculate_importances()
# feature_importance.calculate_cumulative_importance()
# features_to_keep, features_to_discard = feature_importance.find_highly_correlated_features()

# print("Features to keep:", features_to_keep)
# print("Features to discard:", features_to_discard)

# feature_importance.plot_importances()
