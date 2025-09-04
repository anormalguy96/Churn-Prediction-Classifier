# Churn Prediction — UNEC ICT Club

A simple, practical pipeline for predicting customer churn using the Telco dataset.
The notebook walks from raw CSV → cleaning → EDA → preprocessing → models → picking and saving the final pipeline so you can use it on new data.

---

## Modelling and evaluation (what I did)

* Cleaned and inspected the data (converted `TotalCharges` to numeric, stripped whitespace).
* Built a `ColumnTransformer` pipeline:

  * numeric: median imputer → `StandardScaler`
  * categorical: most frequent imputer → `OneHotEncoder(handle_unknown='ignore')`
* Baseline: RandomForest inside the pipeline.
* Tuning: `skopt.BayesSearchCV` to tune RF hyperparameters.
* Compared tuned RF with default XGBoost and CatBoost.
* Picked the best model by ROC-AUC on the test set and saved it.

Example results from one run (your numbers can vary):

* Baseline RF ROC-AUC ≈ 0.816
* Tuned RF ROC-AUC ≈ 0.842
* CatBoost ≈ 0.838, XGBoost ≈ 0.825

---

## Notes & tips

* `TotalCharges` has some blank strings -> converting to numeric creates a few `NaN`; the pipeline imputes them.
* `SeniorCitizen` is a binary flag (0/1); don’t treat it like a continuous outlier-prone column when using IQR.
* Keep the imputers and encoders inside the pipeline — that way the saved model can handle raw inputs.
* If business priorities prefer catching churners, tune for recall (or use class weights / resampling).
* If you want explanations, try SHAP on the final model.

---

## Where to go from here

* Add feature engineering (tenure buckets, interactions, usage ratios).
* Try stacking tuned models for a small performance boost.
* Calibrate probabilities or tune decision thresholds for business needs.
* Wrap the saved pipeline in a small API (Flask/FastAPI) for serving predictions.
