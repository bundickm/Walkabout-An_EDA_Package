# Walkabout

Walkabout is a Python package to rapidly perform common exploratory data analysis on all features of a Pandas DataFrame.

## Installation

Use the package manager test pypi (url here)

```bash
pip install -i https://test.pypi.org/simple/ walkabout
```

## Usage
[Example Notebook](https://colab.research.google.com/drive/1Tufo97ZclCujjPtHhNoXNkjGKloi6y4S)

```python
import walkabout as wa

wa.report.rundown(df) #Display summary statistics including nulls, data types, unqiue values, and shape

wa.plot.univariate_distribution(df) #Plot distribution graphs for all features

wa.plot.numeric_distribution(df) #Display skew, kurtosis, and basic translation of skew value for all numeric features.
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

You may contact me via email at bundickm@gmail.com with suggestions, requests, etc..

## License
[MIT](https://choosealicense.com/licenses/mit/)
