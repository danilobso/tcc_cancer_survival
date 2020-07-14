# Cancer Survival

Repo for my personal wrapper for survival analysis of cancer datasets.

This is a very simple tool to use, with only 5 lines of code, you can have a working predictive model for survival analysis.

## Example of utilization:

### Cleaning the data and training the model
```
sl = SurvivalLib(df,
                 target_col,
                 features_to_delete)

sl.clean_data(columns_to_drop,
              min_survival_months,
              max_survival_months)
sl.process_feature_importance()
sl.brute_force_num_features_c_index()
sl.optimize_hyperparams()
```

### Evaluating the model
```
sl.get_c_index()
sl.predict_risk()
```