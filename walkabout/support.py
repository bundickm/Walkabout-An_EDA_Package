import pandas as pd
import numpy as np

__all__ = ['list_to_string', 'strip_columns', 'outlier_mask', 'trimean',
           'variance_coefficient', 'placehold_to_nan']


'''
Supporting functions for exploratory data analysis
'''


def outlier_mask(feature, inclusive=True):
    '''
    Creates a mask of the outliers using IQR

    Input:
    feature: Pandas Series object containing numeric values
    inclusive: bool, default is True, whether to include values that lie on the
              boundary of becoming an outlier. False will consider the edge
              cases as outliers.

    Output:
    Return a Pandas Series object of booleans where True values correspond
    to outliers in the original feature
    '''
    q1 = feature.quantile(.25)
    q3 = feature.quantile(.75)
    iqr = q3-q1
    mask = ~feature.between((q1-1.5*iqr), (q3+1.5*iqr), inclusive=inclusive)
    return mask


def trimean(feature):
    '''
    Calculate the trimean. Trimean is a measure of the
    center that combines the medians emphasis on center values with the
    midhinge's attention to the extremes.

    Input:
    feature: Pandas Series or DataFrame Object containing numeric values

    Output:
    Return the trimean as a float or an array of floats
    '''
    q1 = feature.quantile(.25)
    q2 = feature.median()
    q3 = feature.quantile(.75)

    return ((q1+2*q2+q3)/4)


def variance_coefficient(feature):
    '''
    Calculate the coefficient of variance

    Input:
    feature: Pandas Series or DataFrame Object containing numeric values

    Output:
    Return the coefficient of variance as a float or an array of floats
    '''

    return (feature.var()/feature.mean())


def list_to_string(list, separator=', '):
    '''
    Helper function to convert lists to string and keep clean code

    Input:
    list: a list
    separator: a string used as the separating value between items
               in the list, default = ', '

    Output:
    Return a string made from list with separator between values.
    '''
    # To-Do: rewrite to convert nested lists to string without lists inside
    return separator.join(str(item) for item in list)


def strip_columns(df):
    '''
    Helper function to remove leading or trailing spaces from
    all values in a dataframe

    Input:
    df: Pandas DataFrame Object

    Output:
    Return a Pandas DataFrame object
    '''
    df = df.copy()

    for col in df.select_dtypes(exclude='number').columns:
        df[col] = df[col].str.strip()

    return df


def placehold_to_nan(df, placeholders=[-1, -999, -9999, 'None', 'none',
                                       'missing', 'Missing', 'Null', 'null',
                                       '?', 'inf', np.inf]):
    '''
    Convert all values in df that are in placeholders to NaN

    Input:
    df: Pandas DataFrame or Series object
    placeholders: a list of values used as placeholders for NaN

    Output
    Return df with all placeholder values fill with NaN
    '''
    return df.replace(placeholders, np.NaN)
