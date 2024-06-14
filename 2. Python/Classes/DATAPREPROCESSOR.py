
import numpy as np

import pandas as pd

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def upsample(self, df, target_column):
        """
        Up-sample the minority class of a DataFrame based on the target column.
        """
        df_minority = df[df[target_column] == df[target_column].value_counts().idxmin()]
        df_majority = df[df[target_column] != df[target_column].value_counts().idxmin()]

        df_minority_upsampled = df_minority.sample(n=len(df_majority), replace=True, random_state=42)
        df_upsampled = pd.concat([df_majority, df_minority_upsampled], axis=0)
        
        return df_upsampled

    def from_quartile_idx(self, quartile):
        """
        Get the index of the quartile in df when creating train/test splits.
        """
        counter = 0
        eighty = np.round(self.df.Ever90.value_counts()[1] * quartile)
        position = 0

        for idx, i in enumerate(self.df.Ever90):
            if i == 1:
                counter += 1 
            if counter == eighty:
                position = idx
                break 

        return position



    def get_split(self, train_quartile=0.7, val_quartile=0.85):
        """
        Get train, validation, and test split based on specified quartiles.
        
        Parameters:
        - train_quartile: the end quartile for the training set (default is 70)
        - val_quartile: the end quartile for the validation set (default is 85)
        """
        print('Binary Split: ' + str(self.df.Ever90.value_counts()))

        # Shuffle data
        self.df = self.df.sample(frac=1, random_state=520).reset_index(drop=True)

        # Determine the split indices based on quartiles
        position_train_end = self.from_quartile_idx(train_quartile)
        position_val_end = self.from_quartile_idx(val_quartile)

        train = self.df.iloc[:position_train_end]
        val = self.df.iloc[position_train_end:position_val_end]
        test = self.df.iloc[position_val_end:]

        # Ensure no same application is in both train and test
        inboth_test_train = test[test.ApplicationID.isin(train.ApplicationID)].ApplicationID.unique()
        train_accounts_notintest = train[~train['ApplicationID'].isin(inboth_test_train)][:len(inboth_test_train)]
        test = test[~test.ApplicationID.isin(inboth_test_train)]
        test = pd.concat([test, train_accounts_notintest])
        train_remaining = train[~train['ApplicationID'].isin(train_accounts_notintest['ApplicationID'])].drop_duplicates()
        inboth_test_train_df = self.df[self.df.ApplicationID.isin(inboth_test_train)]
        train = pd.concat([train_remaining, inboth_test_train_df])

        # Ensure no same application is in both train and validation
        inboth_val_train = val[val.ApplicationID.isin(train.ApplicationID)].ApplicationID.unique()
        train_accounts_notinval = train[~train['ApplicationID'].isin(inboth_val_train)][:len(inboth_val_train)]
        val = val[~val.ApplicationID.isin(inboth_val_train)]
        val = pd.concat([val, train_accounts_notinval])
        train_remaining = train[~train['ApplicationID'].isin(train_accounts_notinval['ApplicationID'])].drop_duplicates()
        inboth_val_train_df = self.df[self.df.ApplicationID.isin(inboth_val_train)]
        train = pd.concat([train_remaining, inboth_val_train_df])

        return train.drop_duplicates(), val.drop_duplicates(), test.drop_duplicates()


    # def get_split(self, quartile1):
    #     """
    #     Get train and test split based on the specified quartile.
    #     """
    #     print('Binary Split: ' + str(self.df.Ever90.value_counts()))

    #     # Shuffle data
    #     self.df = self.df.sample(frac=1, random_state=420).reset_index(drop=True)

    #     # Determine the split index
    #     position1 = self.from_quartile_idx(quartile1)
    #     train, test = self.df.iloc[:position1], self.df.iloc[position1:]

    #     # Ensure no same application is in both train and test
    #     inboth = test[test.ApplicationID.isin(train.ApplicationID)].ApplicationID.unique()

    #     # Take the corresponding number from train and put back to test
    #     train_accounts_notintest = train[~train['ApplicationID'].isin(inboth)][:len(inboth)]

    #     # Remove these from test
    #     test = test[~test.ApplicationID.isin(inboth)]

    #     # Add train accounts not in test to test
    #     test = pd.concat([test, train_accounts_notintest])

    #     # Complete train set without the moved accounts
    #     train_remaining = train[~train['ApplicationID'].isin(train_accounts_notintest['ApplicationID'])].drop_duplicates()

    #     # Creating a DataFrame from the inboth array
    #     inboth_df = self.df[self.df.ApplicationID.isin(inboth)]

    #     # Concatenate the train_remaining DataFrame with the inboth DataFrame
    #     train = pd.concat([train_remaining, inboth_df])

    #     return train, test


    def scaller(self, train, val, test, features):
        """
        Scale the features of train, validation, and test data.
        """
        # Create Xtrain, Ytrain, Xval, Yval, Xtest, Ytest
        Xtrain, Ytrain = train[features], train['Ever90']
        Xval, Yval = val[features], val['Ever90']
        Xtest, Ytest = test[features], test['Ever90']

        # Scale the data
        scaler = StandardScaler()
        Xtrain_scaled = scaler.fit_transform(Xtrain)
        Xval_scaled = scaler.transform(Xval)
        Xtest_scaled = scaler.transform(Xtest)

        return (Xtrain_scaled, Ytrain.values), (Xval_scaled, Yval.values), (Xtest_scaled, Ytest.values)


    def get_split_data_with_upsample_and_scaling(self, quartile1, upsample=True):
        """
        Get split data with optional upsample and scaling.
        """
        train, test = self.get_split(quartile1)
        features = [col for col in self.df.columns if col not in ['Ever90', 'ApplicationID']]

        if upsample:
            train_upsampled = self.upsample(train, 'Ever90')
            print(train_upsampled.Ever90.value_counts())
            return self.scaller(train_upsampled, test, features)
        else:
            return self.scaller(train, test, features)
