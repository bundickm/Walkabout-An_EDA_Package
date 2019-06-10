import unittest
import pandas as pd
import numpy as np
import math
import support


class SupportOutlierMaskTests(unittest.TestCase):
    '''
    Test the outlier_mask function in support.py
    '''
    def test_empty_object(self):
        sample = pd.Series()
        self.assertEqual(list(support.outlier_mask(sample)), [])

    def test_constant_value(self):
        sample = pd.Series([1, 1, 1, 1, 1])
        outcome = pd.Series([False, False, False, False, False])
        self.assertEqual(list(support.outlier_mask(sample)), list(outcome))

    def test_large_outlier(self):
        sample = pd.Series([1, 1, 1, 1, 1000])
        outcome = pd.Series([False, False, False, False, True])
        self.assertEqual(list(support.outlier_mask(sample)), list(outcome))

    def test_small_outlier(self):
        sample = pd.Series([1, .000000001, 1, 1, 1])
        outcome = pd.Series([False, True, False, False, False])
        self.assertEqual(list(support.outlier_mask(sample)), list(outcome))

    def test_negative_outlier(self):
        sample = pd.Series([-10, 1, 1, 1, 1])
        outcome = pd.Series([True, False, False, False, False])
        self.assertEqual(list(support.outlier_mask(sample)), list(outcome))

    def test_steady_step_values(self):
        sample = pd.Series([1, 2, 3, 4, 5])
        outcome = pd.Series([False, False, False, False, False])
        self.assertEqual(list(support.outlier_mask(sample)), list(outcome))

    def test_steady_step_with_outlier(self):
        sample = pd.Series([1, 2, 100, 4, 5])
        outcome = pd.Series([False, False, True, False, False])
        self.assertEqual(list(support.outlier_mask(sample)), list(outcome))


class SupportTrimeanTests(unittest.TestCase):
    '''
    Test the trimean function in support.py
    '''
    def test_empyt_object(self):
        sample = pd.Series([])
        self.assertEqual(math.isnan(support.trimean(sample)), True)

    def test_constant_value(self):
        sample = pd.Series([0, 0, 0, 0, 0])
        self.assertEqual(support.trimean(sample), 0)

    def test_steady_step_values(self):
        sample = pd.Series([10, 20, 30, 40, 50])
        self.assertEqual(support.trimean(sample), 30)

    def test_steady_step_with_outlier(self):
        sample = pd.Series([10, 20, 30, 400, 50])
        self.assertEqual(support.trimean(sample), 32.5)

    def test_irregularly_spaced_sized_values(self):
        sample = pd.Series([10, .0003, -50, 1000, 0])
        self.assertEqual(support.trimean(sample), 2.50015)

    def test_dataframe_input(self):
        sample = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                               'B': [10, 10, 10, 10, 10],
                               'C': [.01, .1, .001, 1, 10]})
        self.assertEqual(list(support.trimean(sample)), [3, 10, .3025])


class SupportVarianceCoefficientTests(unittest.TestCase):
    '''
    Test the variance_coefficient function in support.py
    '''
    def test_constant_value(self):
        sample = pd.Series([1, 1, 1, 1, 1])
        self.assertEqual(support.variance_coefficient(sample), 0.0)

    def test_empty_object(self):
        sample = pd.Series([])
        self.assertEqual(math.isnan(support.variance_coefficient(sample)), True)

    def test_steady_step_values(self):
        sample = pd.Series([10, 20, 30, 40, 50])
        self.assertEqual(
            support.variance_coefficient(sample), 8.333333333333334)

    def test_steady_step_with_outlier(self):
        sample = pd.Series([10, 20, 30, 400, 50])
        self.assertEqual(
            support.variance_coefficient(sample), 274.2156862745098)

    def test_irregularly_spaced_sized_values(self):
        sample = pd.Series([10, .0003, -50, 1000, 0])
        self.assertEqual(
            support.variance_coefficient(sample), 1065.46826704126)

    def test_dataframe_input(self):
        sample = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                               'B': [10, 10, 10, 10, 10],
                               'C': [.01, .1, .001, 1, 10]})
        self.assertEqual(
            list(support.variance_coefficient(sample)),
            [0.8333333333333334, 0.0, 8.586])


class SupportListToStringTests(unittest.TestCase):
    '''
    Test the list_to_string function in support.py
    '''
    def test_default_values(self):
        sample = [1, 2, 3, 4, 5]
        self.assertEqual(
            support.list_to_string(sample), '1, 2, 3, 4, 5')

    def test_mixed_type_in_list(self):
        sample = [1, 'A', True, .05, 'inf']
        self.assertEqual(
            support.list_to_string(sample), '1, A, True, 0.05, inf')

    def test_constant_value(self):
        sample = ['a', 'a', 'a', 'a', 'a']
        self.assertEqual(
            support.list_to_string(sample), 'a, a, a, a, a')

    def test_list_with_object(self):
        df = pd.DataFrame()
        sample = ['a', 1, 2, df, 4]
        self.assertEqual(
            support.list_to_string(sample),
            'a, 1, 2, Empty DataFrame\nColumns: []\nIndex: [], 4')

    def test_nondefault_separator(self):
        sample = [1, 2, 3, 4, 5]
        self.assertEqual(
            support.list_to_string(sample, 'A'),
            '1A2A3A4A5')

    def test_nested_lists(self):
        # This will fail until list_to_string is rewritten
        sample = [1, 2, 3, 4, [5, 5, 5]]
        self.assertEqual(
            support.list_to_string(sample),
            '1, 2, 3, 4, 5, 5, 5')

    def test_nested_lists_with_nondefault_separator(self):
        # This will fail until list_to_string is rewritten
        sample = [1, 2, 3, 4, [5, 5, 5]]
        self.assertEqual(
            support.list_to_string(sample, '!'),
            '1!2!3!4!5!5!5')

    def test_empty_list(self):
        self.assertEqual(support.list_to_string([]), '')


class SupportStripColumns(unittest.TestCase):
    def test_empty_object(self):
        sample = pd.DataFrame()
        self.assertEqual(list(support.strip_columns(sample)), [])

    def test_strip_leftside_space(self):
        sample = pd.DataFrame({'a': [' 1', ' a', ' r']})
        self.assertEqual(
            list(support.strip_columns(sample)['a']),
            ['1', 'a', 'r'])

    def test_strip_rightside_space(self):
        sample = pd.DataFrame({'a': ['1 ', 'a ', 'r ']})
        self.assertEqual(
            list(support.strip_columns(sample)['a']),
            ['1', 'a', 'r'])

    def test_strip_bothsides_space(self):
        sample = pd.DataFrame({'a': [' 1 ', ' a ', ' r ']})
        self.assertEqual(
            list(support.strip_columns(sample)['a']),
            ['1', 'a', 'r'])

    def test_strip_multispace(self):
        sample = pd.DataFrame({'a': ['1  ', '  a ', '  r   ']})
        self.assertEqual(
            list(support.strip_columns(sample)['a']),
            ['1', 'a', 'r'])

    def test_strip_newline(self):
        sample = pd.DataFrame({'a': ['\n1 ', 'a ', 'r \n']})
        self.assertEqual(
            list(support.strip_columns(sample)['a']),
            ['1', 'a', 'r'])

    def test_strip_empty_values(self):
        sample = pd.DataFrame({'a': [' ', '', '\n']})
        self.assertEqual(
            list(support.strip_columns(sample)['a']),
            ['', '', ''])


class PlaceholdToNanTests(unittest.TestCase):
    def test_empty_object(self):
        sample = pd.DataFrame()
        self.assertEqual(list(support.placehold_to_nan(sample)), [])

    def test_default_values(self):
        sample = pd.DataFrame({'a': [-999, -1, '?', 'inf']})
        for i in range(len(sample)):
            val = list(support.placehold_to_nan(sample)['a'])[i]
            self.assertEqual(math.isnan(val), True)

    def test_multiple_features(self):
        sample = pd.DataFrame({'a': [-1,'?','inf'],
                               'b': ['null','none','Missing']})
        for col in sample:
            for i in range(len(sample[col])):
                val = list(support.placehold_to_nan(sample)[col])[i]
                self.assertEqual(math.isnan(val), True)
    
    def test_no_placeholds(self):
         sample = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
         self.assertEqual(list(support.placehold_to_nan(sample)['a']), list(sample['a']))

    def test_nondefault_placeholds(self):
        sample = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        self.assertEqual(math.isnan(list(support.placehold_to_nan(sample, [5])['a'])[4]), True)

    def test_nondefault_placeholds_and_no_placeholds(self):
        sample = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        self.assertEqual(list(support.placehold_to_nan(sample, [6])['a']), list(sample['a']))

if __name__ == '__main__':
    unittest.main()
 