# Mock Interview Loan Classification Project  
- version 1.3 | updated 11.09.2026

This repository contains a reusable tabular machine-learning workflow developed around the loan-approval classification dataset. The current implementation is dataset-specific, but the intended structure separates reusable pipeline behavior from configuration and human-reviewed modeling decisions.

## Current project structure

```text
data/                    Source and prepared datasets
src/
  config.py              Paths, seeds, and runtime settings
  data_config.py         Curated dataset and model configuration
  loans_preprocessing.py Dataset-specific validation and feature preparation
  profiling_utils.py     EDA reports, surface insights, and config drafting
  model_pipeline.py      Splitting, encoding, training, evaluation, and logging
final_loan_analysis.ipynb
                         Checkpointed classification analysis and final review
tests/                   Automated and smoke-test coverage
exports/                 Timestamped progress and evaluation checkpoints
models/                  Model and encoder artifacts
reports/                 Evaluation reports and leaderboards

```

## Intended workflow

The project follows this general sequence:

1. Load and validate the raw dataset.
2. Profile structure, missingness, distributions, and target balance.
3. Generate a draft configuration for human review.
4. Apply dataset-specific cleaning and feature engineering.
5. Establish a baseline model and an initial broad feature set.
6. Evaluate candidate models and feature groups using training-only cross-validation.
7. Select and document the final feature set and model.
8. Fit the selected pipeline on development data.
9. Evaluate once on an untouched final holdout.
10. Save metrics, configuration, checkpoints, model artifacts, and interpretation notes.

The final holdout must not be used for feature selection, model comparison, threshold selection, permutation importance, or ablation decisions.

## Configuration lifecycle

`data_config.py` is a curated, dataset-specific source of truth. It should not be silently overwritten by automated EDA output.

### Draft configuration

`profiling_utils.draft_data_config()` and `write_data_config()` are intended to generate a reviewable `data_config_draft.py` or equivalent draft artifact. Draft generation can infer structural information such as:

- expected columns;
- numeric and categorical roles;
- target column;
- missing-value observations.

Modeling decisions remain human-reviewed, including:

- imputation strategy;
- baseline feature columns;
- encoding map;
- noisy or excluded features;
- validation bounds;
- final model specifications.

Draft generation must never replace an existing curated `data_config.py` automatically.

## Baseline and final model specifications

The project should maintain two explicit configuration layers:

```python
BASELINE_MODEL_SPECS  # initial broad evaluation configuration
MODEL_SPECS           # accepted final configuration
```

`BASELINE_MODEL_SPECS` supports reproducible early evaluation using an initial, broad feature group. It is not necessarily the production or final feature set.

`MODEL_SPECS` is the canonical configuration consumed by the final reusable modeling pipeline. For classification, it is expected to reference the accepted task-specific definitions, such as:

```python
CLASSIFICATION_FEATURE_COLUMNS
CLASSIFICATION_ENCODING_MAP
```

For the current loan review, the accepted candidate is the compact six-feature classification set:  
[Correlation Heatmap](https://files.fishandgiraffe.org/api/public/dl/JYpSyyRT?inline=true)


```text
credit_history
applicant_income
coapplicant_income
loan_amount
married
self_employed
```

The baseline and final specifications should be passed explicitly to reusable pipeline functions where practical. This avoids relying on temporary mutation of a global configuration object and allows future datasets to use the same evaluation process with different column names and feature definitions.

## Promotion process for final configuration

After the baseline-versus-final decision is accepted:

1. Create a versioned backup of `src/data_config.py`.
2. Add or update task-specific final feature definitions.
3. Point the classification entry in `MODEL_SPECS` to the final definitions.
4. Preserve `BASELINE_MODEL_SPECS` for reproducibility.
5. Run syntax checks and pipeline smoke tests.
6. Rerun the final classification workflow from the intended configuration.
7. Save the final metrics, model, configuration state, and checkpoint references.
8. Update notebook comments and project documentation after behavior is verified.


## Notebook evaluation phases

`final_loan_analysis.ipynb` contains two traceable evaluation paths:

1. Historical exploratory analysis using the original broad split and feature analysis.
2. Post-feature-set review using a separate development/holdout split, training-only cross-validation, and one final holdout evaluation.

Historical results and checkpoints remain useful for traceability. They should not be presented as the final unbiased performance estimate when the historical test data was used for feature analysis.

The post-review interpretation should report class-specific behavior, not recall alone. For example, high positive-class recall may coexist with weak negative-class recall or specificity. Accuracy, precision, F1, ROC-AUC, confusion matrix, class balance, and business error costs should be considered together.

### Recorded final evaluation and baseline comparison

The historical baseline is retained as a reference point. Its Logistic Regression result, produced with the original 13-feature workflow and historical split, was:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Historical Logistic Regression | 0.800 | 0.805 | 0.954 | 0.873 | 0.715 |

The canonical six-feature models were evaluated on the corrected clean holdout after feature selection was completed:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Canonical Logistic Regression | 0.789 | 0.788 | 0.969 | 0.869 | 0.731 |
| Canonical Gradient Boosting | 0.800 | 0.797 | 0.969 | 0.875 | 0.675 |
| Canonical Decision Tree | 0.700 | 0.779 | 0.815 | 0.797 | 0.608 |

These results are contextual rather than a controlled experiment because the historical and canonical evaluations use different splits and workflows. The canonical results are the final estimates for this project because the feature set was selected with training-only cross-validation and the final holdout was reserved for one final evaluation.
  
**Further discussion**  
- Logistic Regression provides the strongest overall ranking behavior among the canonical models:   
    - its ROC-AUC of `0.731` exceeds Gradient Boosting (`0.675`) and the Decision Tree (`0.608`).  
- Gradient Boosting has slightly higher holdout accuracy and F1,  
  - but its lower ROC-AUC indicates weaker separation when considering thresholds across the full score range.   
- Considering the small dataset, these values are virtually negligable. 
  
  

The canonical Logistic Regression confusion matrix is:   
`[[8, 17]`  
`[2, 63]]`,  
using rows as actual outcomes and columns as predictions.  
- With approval represented by class `1`,  
  - this corresponds to 63 true approvals, 2 missed approvals,   
  - 17 false approvals, and 8 correctly identified rejections.  

- Its recall of `96.9%` therefore describes approval-class coverage, not error-free decision-making;  
- precision (`78.8%`), specificity (`32.0%`), and the false-approval count must be discussed alongside it.

### Interpretation

The compact model uses six high-value fields rather than the original broad feature group. Five-fold training cross-validation found the compact candidate broadly comparable to the all-feature candidate, while reducing the feature surface and simplifying interpretation. The final holdout was then used once, after selection, to estimate performance.

A **source-completeness** analysis found:  
  - only `73.9%` of the original rows were complete across all six selected fields;  
  - `26.1%` were missing at least one *canonical* value.  
  
This supports a defensible statement that:  
- the model achieved useful performance despite substantial incomplete-data exposure.  
- It **can not establish** that missingness caused lower performance without a separate subgroup analysis.  

## Source completeness context

The final notebook measures completeness against the original raw dataset before cleaning or imputation. Across the six compact classification features,  
- 416 of 563 raw rows were complete (`73.9%`),  
- 147 of 563 rows (`26.1%`) were missing at least one *canonical* feature.

Individual missingness was:

- `credit_history`: `3.9%`
- `applicant_income`: `4.6%`
- `coapplicant_income`: `6.0%`
- `loan_amount`: `5.3%`
- `married`: `3.4%`
- `self_employed`: `6.0%`

These figures establish the amount of incomplete source data present during modeling, and support that the model achieved its reported performance despite **substantial missing-data** exposure and the need for preprocessing or imputation.  
- Improved data collection methods and standards are highly encouraged.
  
---

<br>

## Business Insights from Initial Data Analysis

**Raw dataset**  
[EDA summary.txt](https://files.fishandgiraffe.org/api/public/dl/e0PpwxtC?inline=true)
- 563 records & 13 fields.
- 535 records with known outcome  
  - 384 loans were approved (71.8%)  
  - 151 were rejected (28.2%)  
  - The target is imbalanced toward approvals  
  [loan_status chart](https://files.fishandgiraffe.org/api/public/dl/HMzyV9QQ?inline=true)  
- 25 duplicate rows
- 28 records (5.0%): no loan-status outcome   
  - removed from model training  

**Demographics**  
- male (81.1%), married (64.0%)  
[gender chart](https://files.fishandgiraffe.org/api/public/dl/KmLLa-gS?inline=true)
- no dependents (57.8%)    
  - [dependents chart](https://files.fishandgiraffe.org/api/public/dl/GYWDbEpt?inline=true)
  - [married chart](https://files.fishandgiraffe.org/api/public/dl/QM2qjEGA?inline=true)
- graduate-educated (78.7%)  
[education chart](https://files.fishandgiraffe.org/api/public/dl/Rj5rpeGS?inline=true)
- 'No self-employment' (87.1%)   
[self-employment chart](https://files.fishandgiraffe.org/api/public/dl/ggtW0mpl?inline=true)

**Finances**
- Credit history
  - positive (87.8%)  
  - highly concentrated binary field   
  [credit history histogram](https://files.fishandgiraffe.org/api/public/dl/8w5RaB3s?inline=true)  
- Income (applicant) concentrated 2,500 - 5,000 units (51.4%)  
  - 'monthly' vs 'annual' not specified  
  - values appear to equate to 'annual'
  - but no label = unclear  
  [applicant_income histogram](https://files.fishandgiraffe.org/api/public/dl/GriXHmfn?inline=true)  
  [applicant_income boxplot](https://files.fishandgiraffe.org/api/public/dl/zfbN7EuS?inline=true)
  
- Income (coapplicant) = $0 (43.3%)  
  - likely applicants without a coapplicant 
  - not automatically seen as 'missing'  
  [coapplicant_income histogram](https://files.fishandgiraffe.org/api/public/dl/hQMbqfAA?inline=true)  
  [coapplicant_income boxplot](https://files.fishandgiraffe.org/api/public/dl/x25sjVSF?inline=true)  


**Loans**
- amount concentrated 100 - 200 units (61.4%)  
  [loan_amount histogram](https://files.fishandgiraffe.org/api/public/dl/UN7ekCkY?inline=true)  
  [loan_amount boxplot](https://files.fishandgiraffe.org/api/public/dl/N9QpcmNf?inline=true) 
- '360' month term (86.0%)    
[loan_amount_term histogram](https://files.fishandgiraffe.org/api/public/dl/AMa9ZkUE?inline=true)  
[loan_amount_term boxplot](https://files.fishandgiraffe.org/api/public/dl/YSIEXJV8?inline=true) 
- Property locations  
  - Semiurban (39.1%)
  - Urban (32.3%)  
  - Rural (28.6%)  
  - more balanced than other categorical fields   
    [property_area chart](https://files.fishandgiraffe.org/api/public/dl/q7qjNZO9?inline=true) 

**Missing Data**
- Above insights *ONLY* reflect **'non-missing'** values
- `Missingness` is distributed across multiple fields  
  - not isolated to any column  
- **Dominant** missing fields:  
  - self-employment (6.0%) and coapplicant income (6.0%) each  

**Numeric Value Distribution**  
- `Applicant income`, `coapplicant income`, and `loan amount` are right-skewed  
  - median-based imputation is more defensible than mean imputation.


## Key Project Characteristics  
  
- *Leakage prevention:* the data was split before learned imputation and encoding; encoders were fitted only on training data or training folds.  

- *Reproducibility:* baseline and final model specifications are preserved separately, so the original analysis remains repeatable.  

- *Feature governance:* the six-feature model reduces complexity while retaining performance comparable to the broader feature set.  
 
- *Metric trade-off:* high approval recall comes with lower rejection specificity and 17 false approvals in the Logistic Regression holdout.  

- *Model choice:* Gradient Boosting slightly improves accuracy and F1, but Logistic Regression has stronger ROC-AUC and clearer interpretability.     

- *Dataset scale:* the final holdout contains only 90 records, so results are informative but should not be treated as production-grade certainty.  

- *Operational next step:* threshold selection should reflect the relative business cost of false approvals versus missed approvals.  

- *Data quality:* only 73.9% of original rows were complete across the six selected fields, yet the model achieved useful discrimination after controlled preprocessing.  

- *Further suggestions:* Data collection should be reviewed for opportunities to capture more complete information. A larger dataset is needed to provide a more accurate estimation of accuracy.
<br>  

--- 
## Reproducibility and Artifact Checklist

- [X] README links and image references resolve correctly.
- [X] `final_loan_analysis.ipynb` contains the final documented workflow and saved outputs.
- [X] `src/data_config.py` contains the canonical six-feature classification configuration.
- [X] `BASELINE_MODEL_SPECS` remains available for historical reproducibility.
- [X] Focused automated tests pass.
- [X] Notebook code cells compile successfully.
- [X] Final metrics in the README match the notebook outputs.
- [X] No credentials, raw data, private identifiers, or internal-only files are published.
- [X] Excluded folders and files are covered by `.gitignore`.
- [X] Aggregate figures are safe for the intended audience and do not expose row-level data.

---

<br>

This README is an outline and foundation. Dataset-specific details, exact acceptance thresholds, model registry conventions, and deployment procedures should be refined as the workflow matures.


