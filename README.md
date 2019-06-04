# EDA

EDA is a Python package to rapidly perform common exploratory data analysis on all features of a Pandas DataFrame.

## Installation

Use the package manager test pypi (url here)

```bash
pip install testpypi url eda
```

## Usage

```python
import eda

eda.report.rundown(df) #Display summary statistics including nulls, data types, unqiue values, and shape

eda.plot.univariate_distribution(df) #Plot distribution graphs for all features

eda.plot.numeric_distribution(df) #Display skew, kurtosis, and basic translation of skew value for all numeric features.
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

You may contact me via email at bundickm@gmail.com with suggestions, requests, etc..

## License
[MIT](https://choosealicense.com/licenses/mit/)
