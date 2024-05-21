import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def upsample(self, df, target_column): 
        """
        Up-sample the minority class of a DataFrame based on the target column.

        Parameters:
        - df: DataFrame, the DataFrame to upsample
        - target_column: string, the name of the target column

        Returns:
        - DataFrame with balanced classes
        """
        df_minority = df[df[target_column] == df[target_column].value_counts().idxmin()]
        df_majority = df[df[target_column] != df[target_column].value_counts().idxmin()]

        df_minority_upsampled = df_minority.sample(n=len(df_majority), replace=True, random_state=42)
        df_upsampled = pd.concat([df_majority, df_minority_upsampled], axis=0)
        
        return df_upsampled

    def from_quartile_idx(self, quartile):
        """
        Get the index of the quartile in df when creating train/test splits.

        Parameters:
        - quartile: float, the quartile to split on

        Returns:
        - position: int, the index position in the DataFrame
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

    def get_split_data_with_upsample_and_scaling(self, quartile1):
        """
        Get upsampled data where the split is based on a specified quartile to harmonize distribution,
        and then scale the data.

        Parameters:
        - quartile1: float, the quartile to split on

        Returns:
        - Tuple containing (Xtrain, Ytrain), (Xtest, Ytest), and train_upsampled
        """
        print('Binary Split: ' + str(self.df.Ever90.value_counts()))

        # Shuffle the DataFrame
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Determine the split index
        position1 = self.from_quartile_idx(quartile1)
        train, test = self.df.iloc[:position1], self.df.iloc[position1:]

        # Upsample only the training data
        train_upsampled = self.upsample(train, 'Ever90')
        print(train_upsampled.Ever90.value_counts())

        # Create Xtrain, Ytrain, Xtest, Ytest
        Xtrain, Ytrain = train_upsampled.drop(columns='Ever90'), train_upsampled['Ever90']
        Xtest, Ytest = test.drop(columns='Ever90'), test['Ever90']

        # Scale the data
        scaler = StandardScaler()
        Xtrain_scaled = scaler.fit_transform(Xtrain)
        Xtest_scaled = scaler.transform(Xtest)

        return (Xtrain_scaled, Ytrain.values), (Xtest_scaled, Ytest.values), train_upsampled
