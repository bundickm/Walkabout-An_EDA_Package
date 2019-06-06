import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from matplotlib.colors import LinearSegmentedColormap


__all__ = ['univariate_distribution', 'bivariate_categorical_distribution',
           'probability', 'correlation_heatmap', 'null_correlation_heatmap',
           'missingness_map']


def univariate_distribution(df, cols=5, figsize=(20,15),
                            hspace=0.5, wspace=0.5):
    '''
    plot the distribution of all features in a dataframe
    original function found here:
    https://github.com/dformoso/sklearn-classification/blob/master/Data%20Science%20Workbook%20-%20Census%20Income%20Dataset.ipynb

    Input:
    df: Pandas DataFrame object
    cols: number of graphs to display per row
    figsize: tuple of floats representing height and width of the plots
    hspace: the amount of height reserved for space between subplots
    wspace: the amount of width reserved for space between subplots

    Output:
    Display n graphs to the screen, where n is the number of features in df
    '''
    # plot settings
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=wspace, hspace=hspace)
    rows = ceil(float(df.shape[1]) / cols)

    # plot graphs, graph type determined by categoric or numeric feature
    for i, column in enumerate(df.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if df.dtypes[column] == np.object:
            g = sns.countplot(y=column, data=df)
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            plt.xticks(rotation=25)
        else:
            g = sns.distplot(df[column])
            plt.xticks(rotation=25)


def bivariate_categorical_distribution(df, hue, cols=5, figsize=(20, 15),
                                       hspace=0.2, wspace=0.5):
    '''
    Plot a count of the categories from each categorical feature split by hue
    original function found here:
    https://github.com/dformoso/sklearn-classification/blob/master/Data%20Science%20Workbook%20-%20Census%20Income%20Dataset.ipynb

    Input:
    df: Pandas DataFrame object
    hue: a categorical feature, likely the target feature
    cols: number of graphs to display per row
    figsize: tuple of floats representing height and width of the plots
    hspace: the amount of height reserved for space between subplots
    wspace: the amount of width reserved for space between subplots

    Output:
    Display n graphs to the screen, where n is the number of features in df
    '''
    # plot settings
    df = df.select_dtypes(include=[np.object])
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=wspace, hspace=hspace)
    rows = ceil(float(df.shape[1]) / cols)

    # plot each feature's distribution against hue
    for i, column in enumerate(df.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        g = sns.countplot(y=column, hue=hue, data=df)
        substrings = [s.get_text()[:10] for s in g.get_yticklabels()]
        g.set(yticklabels=substrings)



def residuals(df, target, cols=3, figsize=(10, 15), hspace=1, wspace=1):
    '''
    Create residual plots for all numeric features. Useful for
    seeing heteroscedasticity.

    Input:
    df: Pandas DataFrame object
    target: string of the target feature in df
    cols: number of graphs to display per row
    figsize: tuple of floats representing height and width of the plots
    hspace: the amount of height reserved for space between subplots
    wspace: the amount of width reserved for space between subplots

    Output:
    Display n graphs to the screen, where n is the number of
    numeric features in df
    '''
    X = df.drop(target, axis=1).select_dtypes(include='number')
    y = df[target]

    # plot settings
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=wspace, hspace=hspace)
    rows = ceil(float(df.shape[1]) / cols)

    # plot graphs
    for i, column in enumerate(X.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        sns.residplot(X[column], y, line_kws=dict(color='r'), lowess=True)
        ax.set_title(column)


def correlation_heatmap(df, figsize=(5, 5), annot=True):
    '''
    Heatmap of feature correlations of df

    Input:
    df: Pandas DataFrame object
    figsize: tuple of the height and width of the heatmap
    annot: bool, whether to display values inside the heatmap

    Output:
    display heatmap of the feature correlations of df
    '''
    corr = df.corr()

    # generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # plot it
    plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=annot, square=True)


def null_correlation_heatmap(df, figsize=(5, 5), annot=True):
    '''
    heatmap of correlation heatmap of nulls

    Input:
    df: Pandas DataFrame object
    figsize: tuple of the height and width of the heatmap
    annot: bool, whether to display values inside the heatmap

    Output:
    display heatmap of the correlations of nulls in df
    '''
    null_corr = df.isnull().corr()

    # generate a mask for the upper triangle
    mask = np.zeros_like(null_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # plot it
    plt.subplots(figsize=figsize)
    sns.heatmap(null_corr, mask=mask, annot=annot, square=True)


def missingness_map(df, figsize=(5,5), data_name='DataFrame'):
    '''
    graph of the location of all missing values in a dataframe

    Input:
    df: Pandas DataFrame object
    data_name: string for the title of the graph

    Output
    graph of the location of all missing values in a dataframe
    '''
    # map where nulls are in the dataframe
    null_map = df.isnull()

    # create a binary colormap
    myColors = ((0, 0, 0, 1), (1, 1, 1, 1))
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

    # use sns.heatmap and basic cleanup
    ax = sns.heatmap(null_map, vmin=0, vmax=1, cmap=cmap,)
    ax.set(yticks=[])
    ax.set_ylabel('Observations\n(Descending Order)')
    ax.set_xlabel('Features')
    plt.title(f'Missingness Map for {data_name}')

    # Binary color bar with labels
    ax.collections[0].colorbar.set_ticks([.75, .25])
    ax.collections[0].colorbar.set_ticklabels(['Null', 'Not Null'])

    # display the graph
    plt.subplots(figsize=figsize)
    plt.show()
