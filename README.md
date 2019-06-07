# Walkabout

Walkabout is a Python package to rapidly perform common exploratory data analysis on all features of a Pandas DataFrame.

### What's in a name?
A Walkabout is a term from Australian Aborigines and is a journey of discovery and of self. In earlier times, it was known as a rite of passage where male Aborigine adolescents would embark on a journey into the wilderness, sometimes for as long as six months.

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
If you are interested in contributing, have feature requests, or bugs - please reach out to me.

You may contact me via email at bundickm@gmail.com.

## License
[MIT](https://choosealicense.com/licenses/mit/)
