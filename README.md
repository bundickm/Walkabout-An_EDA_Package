# Walkabout

Walkabout is a Python package to rapidly perform common exploratory data analysis on all features of a Pandas DataFrame. A single line of code is all you need to plot the distribution of all features, view detailed summary stats, find which feature interactions are likely to be important, and more. Walkabout is all about quick, easy discovery about your entire dataframe.

### What's in a name?
A Walkabout is a term from Australian Aborigines and is a journey of discovery and of self. In earlier times, it was known as a rite of passage where male Aborigine adolescents would embark on a journey into the wilderness, sometimes for as long as six months.

## Installation

Use the package manager test.pypi (https://test.pypi.org/project/walkabout/)

```bash
pip install -i https://test.pypi.org/simple/ walkabout
```

## Usage
If you would like to see all the functionality available in walkabout, check out the example notebook:

[Example Notebook](https://colab.research.google.com/drive/1Tufo97ZclCujjPtHhNoXNkjGKloi6y4S)

```python
import walkabout as wa

# Display summary statistics including nulls, data types, unqiue values, and shape
wa.report.rundown(df)
```
![](https://raw.githubusercontent.com/bundickm/walkabout/blob/master/images/rundown.png)

```python
# Plot distribution graphs for all features
wa.plot.univariate_distribution(df)
```
![](https://raw.githubusercontent.com/bundickm/walkabout/master/images/univariate_distribution.png)

```python
# Display skew, kurtosis, and basic translation of skew value for all numeric features.
wa.plot.numeric_distribution(df)
```
![](https://raw.githubusercontent.com/bundickm/walkabout/master/images/numeric_distribution.png)

```python
# Plot box plots for all features, either univariate or bivariate
wa.plot.boxplot(df, 'age')
```
![](https://raw.githubusercontent.com/bundickm/walkabout/master/images/boxplot.png)

```python
# Quickly find which features might be important
wa.report.simple_feature_importance(X, y, model='clas')
```
![](https://raw.githubusercontent.com/bundickm/walkabout/master/images/feature_importance.png)

## Contributing
If you are interested in contributing, have feature requests, or bugs - please reach out to me.

You may contact me via email at bundickm@gmail.com.

## License
[MIT](https://choosealicense.com/licenses/mit/)
