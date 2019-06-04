'''
walkabout - a package for quick exploratory analysis on all features in a dataframe
'''

import pandas as pd
import numpy as np
from . import report, plot, support

ONES = pd.DataFrame(np.ones(10))