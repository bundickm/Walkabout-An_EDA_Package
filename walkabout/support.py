import pandas as pd


__all__ = ['list_to_string', 'strip_columns', 'outlier_mask', 'trimean',
           'variance_coefficient']


'''
Supporting functions for exploratory data analysis, eventually to be
broken into Support and Clean
'''


def _placeholders_present(column, placeholders):
    '''
    Return a list of values that are both in column and placeholders

    Input:
    column: Pandas Series object
    placeholders: a list of values commonly used in place of null

    Output:
    Return a list of values that are both in column and placeholders
    '''
    p_holds = []
    for item in placeholders:
        if len(column.isin([item]).unique()) == 2:
            p_holds.append(item)
    return list_to_string(list(set(p_holds)))


def _null_rec_lookup(null_percent, placeholders=False):
    '''
    Recommend course of action for handling nulls based on
    findings from Report.nulls

    Input:
    null_percent: float, percent of a column that is null
    placeholders: bool, whether the column contains placeholders

    Output:
    Return a string recommendation
    '''
    # Temp Recommendations - Switch to MCAR, MAR, MNAR assessment and recs
    # https://www.youtube.com/watch?v=2gkw2T5jAfo&feature=youtu.be
    # https://stefvanbuuren.name/fimd/sec-MCAR.html
    if placeholders:
        return 'Possible Placeholders: Replace and rerun nulls report.'
    elif null_percent == 100:
        return 'Empty Column: Drop column'
    elif null_percent >= 75:
        return 'Near Empty Column: Create binary feature or drop'
    elif null_percent >= 25:
        return 'Partially Filled Column: Assess manually'
    elif null_percent > 0:
        return 'Mostly Filled Column: Impute values'
    else:
        return ''


def _skew_translation(skew):
    '''
    Gives a summary phrase for a skew

    Input:
    skew: float

    Output:
    Return a string  of skew's corresponding summary phrase
    '''
    if (skew < -1) or (skew > 1):
        return 'Highly Skewed'
    if (-1 <= skew <= -.5) or (.5 <= skew <= 1):
        return 'Moderately Skewed'
    else:
        return 'Approximately Symmetric'


def list_to_string(list):
    '''
    Helper function to convert lists to string and keep clean code

    Input:
    list: a list

    Output:
    Return a string made from list with comma and space separator
    between values.
    '''
    return ', '.join(str(item) for item in list)


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
