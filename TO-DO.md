# To Do List
List of tasks that still need to be completed for walkabout

## Feature Additions
- Create simple design guidelines documentation
- Expand readme with example reports and images
- Statistical tests for MCAR (Little's T-Test)
- Update nulls report recommendations using Little's T-Test
- Update nulls report recommendations with details beyond "impute values" or "assess manually", such as "Impute with KNN"
- Rewrite list_to_string() to convert nested lists into single string without lists
- Reports of time series analysis
- Additional functions for measures of centrality
- Additional functions for variance
- Image analysis

## Testing
### report.py
- All functions require testing
### plot.py
- All functions require testing
### support.py
- test_nested_lists() will fail until list_to_string() is rewritten
- test_nested_lists_with_nondefault_separator() will fail until list_to_string() is rewritten

## Bugs
- None currently identified

## Possible Extensions
- Addition of imputation and cleaning methods
- Addition of common and simple feature engineering such as datetime into Month, Day, and Year columns, or encoding common categoricals like states.
