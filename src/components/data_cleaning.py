import pandas as pd

class DataCleaning():
    def __init__(self):
        pass
        
    def iqr_outliers(self, df, columns):
        outliers_list = []

        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outliers_list.append(outliers)

        all_outliers_df = pd.concat(outliers_list).drop_duplicates()

        return all_outliers_df
    
    def drop_subset_rows(self, df, subset_df):
        subset_indices = subset_df.index
        
        df_cleaned = df.drop(index=subset_indices)
        
        return df_cleaned


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
        df[column].fillna(df[column].mean(), inplace=True)