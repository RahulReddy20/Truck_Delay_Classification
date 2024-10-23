import pandas as pd

class DataCleaning():
    def __init__(self):
        pass
        
    def iqr_outliers(self, df, columns):
        """
        Detect outliers in the specified columns of a DataFrame using the IQR method
        and return all the outlier rows combined as a single DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list): List of column names to check for outliers.

        Returns:
        pd.DataFrame: A DataFrame containing all the outlier rows.
        """
        outliers_list = []

        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Extract the outliers for this column
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outliers_list.append(outliers)

        # Concatenate all outlier DataFrames and drop duplicates
        all_outliers_df = pd.concat(outliers_list).drop_duplicates()

        return all_outliers_df
    
    def drop_subset_rows(self, df, subset_df):
        """
        Drop rows from the original DataFrame (df) that are present in the subset DataFrame (subset_df).

        Parameters:
        df (pd.DataFrame): The original DataFrame.
        subset_df (pd.DataFrame): The subset DataFrame containing the rows to be dropped.

        Returns:2114
        
        pd.DataFrame: A new DataFrame with the subset rows dropped.
        """
        # Find the index of the rows in subset_df
        subset_indices = subset_df.index
        
        # Drop those indices from the original DataFrame
        df_cleaned = df.drop(index=subset_indices)
        
        return df_cleaned

    # Example Usage
    # cleaned_df = drop_subset_rows(city_weather_df, outliers_df)

    def categorize_time_of_day(self, hour):
        time_periods = {
            (300, 600): 'Early Morning',
            (600, 900): 'Morning',
            (900, 1200): 'Late Morning',
            (1200, 1500): 'Afternoon',
            (1500, 1800): 'Late Afternoon',
            (1800, 2100): 'Evening',
            (2100, 2400): 'Night',
            (0, 300): 'Late Night'
        }
        
        for (start, end), period in time_periods.items():
            if start <= hour < end:
                return period
        return 'Invalid Hour'

    def fill_missing_with_mean(self, df, column):
        """
        Fill missing values in a specified column of a DataFrame with the column's mean.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the column with missing values.
        column (str): The column name where missing values need to be filled.

        Returns:
        None: The function modifies the DataFrame in place.
        """
        df[column].fillna(df[column].mean(), inplace=True)