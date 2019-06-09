import pandas as pd
import numpy as np
from math import ceil
from tabulate import tabulate
from . import support


__all__ = ['nulls', 'type_and_unique', 'rundown', 'assess_categoricals', 
           'numeric_distribution']


def nulls(df, placeholders=[-1, -999, -9999, 0, 'None', 'none',
                            'missing', 'Missing', 'Null', 'null',
                            '?', 'inf', np.inf]):
    '''
    Report null distribution, any possible placeholders, and
    simple recommendations

    Input:
    df: Pandas DataFrame object
    placeholders: list of common placeholder values used in place of null.
                  Report.nulls() is case sensitive ('none' != 'None')

    Output:
    Print report to screen
    '''
    null_count = df.isnull().sum().loc
    total = len(df)
    headers = ['Column', 'Nulls', '%Null', 'Placeholders', 'Recommendation']
    table = []

    # Iterate through each column and append null details to table
    for column in df.columns:
        calc = null_count[column]/total*100
        null_per = str(calc)+'%'
        p_hold = _placeholders_present(df[column], placeholders)
        rec = _null_rec_lookup(calc, p_hold)
        table.append([column, null_count[column], null_per, p_hold, rec])

    # output with tabulate library
    print(tabulate(table, headers))


def _describe(df):
    '''
    Simple mod to Pandas.DataFrame.describe() to support Reports.rundown

    Input:
    df: Pandas DataFrame object

    Output:
    Print report to screen
    '''
    headers = ['Column'] + list(df.describe()[1:].T)
    table = df.describe()[1:].T.reset_index().to_numpy()

    # output with tabulate library
    print(tabulate(table, headers))


def type_and_unique(df, unq_limit=10):
    '''
    Report data type of all features, number of unique values, and
    some of those values

    Input:
    df: Pandas DataFrame object
    unq_limit: number of unique items from each feature to display
               if unique items is less than unq_limit then all
               items are displayed

    Output:
    Print report to screen
    '''
    cols = df.columns
    d_types = list(df.dtypes)
    num_unique = list(df.nunique())
    table = []
    headers = ['Column', 'Type', 'nUnique', 'Unique Values']

    for i in range(len(cols)):
        unique_vals = support.list_to_string(
                      list(df[cols[i]].unique()[:unq_limit]))
        if num_unique[i] == 1:
            unique_vals += ' WARNING: CONSTANT VALUE'
        elif (len(list(df[cols[i]].unique())) > unq_limit):
            unique_vals += '...'
        table.append([cols[i], str(d_types[i]), num_unique[i], unique_vals])
    print(tabulate(table, headers))


def rundown(df, include_shape=True, include_describe=True,
            include_nulls=True, include_types_uniques=True):
    '''
    Report giving an overview of a dataframe

    Input:
    df: Pandas DataFrame object

    Output:
    Print report to screen
    '''
    if include_shape is True:
        print('DataFrame Shape')
        print(f'Rows: {df.shape[0]}    Columns: {df.shape[1]}')
        print()
    if include_describe is True:
        _describe(df)
        print()
    if include_nulls is True:
        nulls(df)
        print()
    if include_types_uniques is True:
        type_and_unique(df)


def assess_categoricals(df, low_thresh=.05, high_thresh=.51,
                        return_low_violators=False):
    '''
    Report for categorical features, highlighting labels in a feature
    that are the majority or extreme minority classifiers

    Input:
    df: Pandas DataFrame object
    low_thresh: float minimum percent distribution desired before binning
    high_thresh: float max percent distribution for majority classifiers
    return_low_violators: bool, if true, include labels below low_thresh
                          as part of report

    Output:
    Print report to screen
    '''
    cols = df.select_dtypes(exclude='number').columns
    headers = ['Feature', '# Below Thresh', 'nUnique', 'High Thresh Violators']
    if return_low_violators is True:
        headers.append('Low Thresh Violators')
    table = []

    # iterate over all features
    for feature in cols:
        val_counts = df[feature].value_counts(normalize=True)
        low_thresh_count = 0
        low_thresh_violators = []
        high_thresh_violators = []

        # count and record values below low_thresh, and above high_thresh
        for label in val_counts.index:
            if val_counts[label] < low_thresh:
                low_thresh_count += 1
                low_thresh_violators.append(label)
            elif val_counts[label] > high_thresh:
                high_thresh_violators.append(label)

        # append to table based on whether we are returning low_violators
        if return_low_violators is True:
            table.append([feature, low_thresh_count, len(val_counts),
                          support.list_to_string(high_thresh_violators),
                          support.list_to_string(low_thresh_violators)])
        else:
            table.append([feature, low_thresh_count, len(val_counts),
                          support.list_to_string(high_thresh_violators)])

    # output with tabulate library
    print(tabulate(table, headers))


def numeric_distribution(df):
    '''
    Report the skew and excess kurtosis of all numeric features in a dataframe

    Excess Kurtosis is the kurtosis of the feature minus the kurtosis of
    the normal distribution(kurtosis = 3)

    Input:
    df: Pandas DataFrame object

    Output:
    Print report to the screen
    '''
    headers = ['Feature', 'Skew', 'Skew Meaning', 'Excess Kurtosis']
    table = []
    cols = df.select_dtypes(include='number').columns

    for col in cols:
        skew = df[col].skew()
        table.append([col, skew, _skew_translation(skew),
                     (df[col].kurtosis()-3)])

    print(tabulate(table, headers))


def high_correlations(df, threshold=.7):
    '''
    Report correlations in df that exceed the threshold.

    Input:
    df: Pandas DataFrame object
    threshold: float, default is .7, range should be between [-1, 1],

    Output:
    Print report to the screen.
    '''

    table = []
    headers = ['Feature 1', 'Feature 2', 'Value']
    corr_df = df.corr()
    columns = corr_df.columns

    for i in range(len(corr_df)):
        for j in range(i+1, len(corr_df)):
            if ((corr_df.iloc[i, j]**2) > (threshold**2)):
                table.append([columns[i], columns[j], corr_df.iloc[i, j]])
    print(tabulate(table, headers))
    print('\nThreshold:', threshold)


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
    return support.list_to_string(list(set(p_holds)))


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
    # Recommendation Change - Include MCAR once Little's T-test added
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
