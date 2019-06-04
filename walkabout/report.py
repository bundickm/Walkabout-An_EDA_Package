import pandas as pd
import numpy as np
from math import ceil
from tabulate import tabulate
from . import support


def nulls(df, placeholders=[-1, -999, -9999, 'None', 'none', 'missing',
                            'Missing', 'Null', 'null', '?', 'inf']):
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
        p_hold = support._placeholders_present(df[column], placeholders)
        rec = support._null_rec_lookup(calc, p_hold)
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
        if (len(list(df[cols[i]].unique())) > unq_limit):
            unique_vals += '...'

        table.append([cols[i], str(d_types[i]), num_unique[i], unique_vals])
    print(tabulate(table, headers))


def rundown(df, include_shape = True, include_describe = True,
            include_nulls = True, include_types_uniques = True):
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
                          high_thresh_violators, low_thresh_violators])
        else:
            table.append([feature, low_thresh_count, 
                          len(val_counts), high_thresh_violators])

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
        table.append([col, skew, support._skew_translation(skew),
                     (df[col].kurtosis()-3)])

    print(tabulate(table, headers))
