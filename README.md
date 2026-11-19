# Retail Sales Dynamics 🛒

A comprehensive **Time Series Forecasting and Machine Learning project for analyzing and predicting daily retail sales revenue** using Python, Scikit-learn, Statsmodels, and MLflow.

The project focuses on understanding historical retail transactions, analyzing revenue patterns, preparing chronological daily revenue data, engineering temporal and historical features, developing and comparing forecasting models, evaluating prediction performance, and explaining forecasting behavior through feature importance and ensemble analysis.

The forecasting framework predicts daily net merchandise revenue over the next **7 observed trading days**, combining statistical forecasting, machine learning, and ensemble approaches.

---

## 🌟 Features

### Core Features

- **Data Understanding & Cleaning:** Inspection of retail transactions, dataset structure, missing values, duplicates, quantities, prices, and invoice characteristics.
- **Exploratory Data Analysis:** Investigation of transaction activity, revenue distributions, daily sales trends, trading-day patterns, and seasonal behavior.
- **Time Series Preparation:** Conversion of transaction-level records into a chronological daily net merchandise revenue series.
- **Feature Engineering:** Development of calendar, historical lag, rolling-window, and revenue-change features.
- **Forecasting Models:** Development and comparison of statistical forecasting, machine learning, and baseline approaches.
- **Hyperparameter Optimization:** Model configuration tuning using chronological validation.
- **Ensemble Forecasting:** Integration of AutoReg, SARIMA, and Extra Trees predictions through equal-weight averaging.
- **Temporal Validation:** Chronological model validation and selection using forecasting performance metrics.
- **Model Evaluation:** Assessment using MAE, RMSE, and R².
- **Rolling-Origin Forecasting:** Evaluation across 55 forecast origins and seven trading-day horizons.
- **Forecast Error Analysis:** Investigation of prediction errors across forecasting horizons and revenue levels.
- **Model Explainability:** Analysis of Extra Trees feature importance, permutation importance, statistical model parameters, and ensemble disagreement.
- **Experiment Tracking:** Management of model experiments, configurations, performance metrics, and artifacts using MLflow.
- **Business Insights:** Identification of revenue patterns, forecasting limitations, and opportunities for improved short-term retail planning.

---

## 🗂 Project Structure

The project is organized as follows:

```text
Retail-Sales-Dynamics/
│
├── data/
│   ├── raw/                                      # Original UCI Online Retail dataset
│   └── processed/                                # Prepared forecasting datasets
│       ├── daily_revenue.parquet
│       ├── forecast_features_train.parquet
│       └── forecast_features_test.parquet
│
├── notebooks/
│   ├── 01_data_understanding.ipynb               # Dataset inspection
│   ├── 02_exploratory_data_analysis.ipynb        # Exploratory Data Analysis
│   ├── 03_data_preparation.ipynb                 # Cleaning and chronological splitting
│   ├── 04_feature_engineering.ipynb              # Temporal and historical features
│   ├── 05_model_development.ipynb                # Forecasting model experimentation
│   ├── 06_model_evaluation.ipynb                 # Rolling-origin evaluation and error analysis
│   └── 07_model_explainability.ipynb             # Feature importance and ensemble interpretation
│
├── models/
│   ├── final_autoreg.pkl                        # Selected AutoReg model
│   ├── final_sarima.pkl                         # Selected SARIMA model
│   └── final_extra_trees.joblib                 # Seven horizon-specific Extra Trees models
│
├── artifacts/
│   ├── forecast_feature_config.json
│   ├── final_forecasting_configuration.json
│   ├── test_model_performance.csv
│   ├── horizon_wise_performance.csv
│   ├── rolling_test_predictions.parquet
│   ├── forecast_error_analysis.parquet
│   ├── revenue_level_error_summary.csv
│   ├── validation_test_comparison.csv
│   ├── model_evaluation_summary.json
│   ├── extra_trees_feature_importance.csv
│   ├── aggregated_feature_importance.csv
│   ├── horizon_feature_importance.csv
│   ├── horizon_1_permutation_importance.csv
│   ├── feature_importance_method_comparison.csv
│   ├── autoreg_coefficients.csv
│   ├── sarima_parameters.csv
│   ├── statistical_model_comparison.csv
│   ├── ensemble_component_summary.csv
│   ├── ensemble_horizon_disagreement.csv
│   └── model_explainability_summary.json
│
├── reports/
│   └── figures/                                  # EDA, evaluation, and explainability visualizations
│
├── requirements.txt                              # Python dependencies
├── README.md                                     # Project documentation
└── .gitignore                                    # Git ignored files
```

**Note:** The structure highlights the principal reusable artifacts. Additional experiment outputs and intermediate files may be generated during notebook execution. Some trained statistical models are refitted at each rolling forecast origin rather than relying exclusively on their saved initial model objects.

---

## 🏗️ Project Architecture

```text
+--------------------------------------+
|              DATA LAYER              |
|                                      |
|       UCI Online Retail Dataset      |
|       Transaction-Level Records      |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|          DATA UNDERSTANDING          |
|                                      |
|  - Dataset inspection                |
|  - Missing values                    |
|  - Duplicate analysis                |
|  - Invoice characteristics           |
|  - Quantity and price validation     |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|              EDA LAYER               |
|                                      |
|  - Transaction characteristics       |
|  - Daily revenue distributions       |
|  - Historical revenue trends         |
|  - Trading-day patterns              |
|  - Seasonality and variability       |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|           DATA PREPARATION           |
|                                      |
|  - Transaction cleaning              |
|  - Revenue calculation               |
|  - Daily revenue aggregation         |
|  - Chronological ordering            |
|  - Train-test splitting              |
|  - Processed artifact creation       |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|          FEATURE ENGINEERING         |
|                                      |
|  - Origin calendar features          |
|  - Target-date calendar features     |
|  - Historical revenue lags           |
|  - Rolling revenue statistics        |
|  - Revenue change indicators         |
|  - Seven-horizon target creation     |
|  - Leakage-aware feature creation    |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|           MODEL DEVELOPMENT          |
|                                      |
|  - Baseline forecasting              |
|  - Statistical forecasting           |
|  - Machine learning regression       |
|  - Chronological validation          |
|  - Hyperparameter optimization       |
|  - Voting and stacking ensembles     |
|  - MLflow experiment tracking        |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|           MODEL EVALUATION           |
|                                      |
|  - Chronological holdout testing     |
|  - Rolling-origin forecasting        |
|  - Seven-horizon evaluation          |
|  - MAE, RMSE, R²                     |
|  - Revenue-level error analysis      |
|  - Forecast bias diagnostics         |
|  - Validation-test comparison        |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|          MODEL EXPLAINABILITY        |
|                                      |
|  - Extra Trees feature importance    |
|  - Permutation importance            |
|  - Horizon-specific interpretation   |
|  - AutoReg coefficient analysis      |
|  - SARIMA parameter analysis         |
|  - Ensemble component performance    |
|  - Forecast disagreement analysis    |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|            BUSINESS INSIGHTS         |
|                                      |
|  - Daily revenue patterns            |
|  - Trading-cycle seasonality         |
|  - Short-term revenue forecasting    |
|  - Revenue planning support          |
|  - Forecast reliability              |
|  - Model limitations                 |
+--------------------------------------+
```

---

## 🛠 Tech Stack

- **Language:** Python 3.11
- **Environment:** Jupyter Notebook, VS Code
- **Libraries:**
  - **Data Manipulation:** Pandas, NumPy
  - **Visualization:** Matplotlib, Seaborn
  - **Machine Learning:** Scikit-learn
  - **Statistical Forecasting:** Statsmodels
  - **Model Evaluation:** Scikit-learn metrics
  - **Experiment Tracking:** MLflow
  - **Artifact Persistence:** Joblib, Pickle, JSON, CSV, Parquet
- **Version Control:** Git & GitHub

---

## 📊 Dataset Overview

The project uses the **Online Retail Dataset**, available from the UCI Machine Learning Repository.

The original dataset contains **541,909 transaction records** and **8 variables**, describing retail invoices, products, transaction quantities, prices, customers, and countries.

### Dataset Features

```text
InvoiceNo
StockCode
Description
Quantity
InvoiceDate
UnitPrice
CustomerID
Country
```

### Target Variable

**`Daily_Revenue`**

The target represents daily net merchandise revenue calculated from eligible retail transaction records.

Transaction-level revenue is calculated as:

\[
\text{Line Revenue} = \text{Quantity} \times \text{UnitPrice}
\]

Eligible transaction revenue values are aggregated by trading date to construct the daily time series.

The objective is to forecast revenue for the next **7 observed trading days** using historical revenue information and calendar characteristics available at each forecast origin.

**Important:** The forecasting horizon is defined in observed trading days rather than consecutive calendar days. The framework assumes future trading dates are known when constructing target-date calendar features.

### Dataset Summary

| Component | Value |
|---|---:|
| Original transaction records | 541,909 |
| Original variables | 8 |
| Cleaned transaction records | 538,998 |
| Total cleaned net merchandise revenue | 9,793,397.24 |
| Daily time series observations | 305 |
| Daily series start date | December 1, 2010 |
| Daily series end date | December 9, 2011 |
| Training observations | 244 trading days |
| Testing observations | 61 trading days |
| Forecast horizon | 7 trading days |
| Training forecast origins | 207 |
| Testing forecast origins | 55 |
| Predictor features per Extra Trees horizon | 48 |
| Total feature dataset columns | 87 |

### Daily Revenue Series

The cleaned transaction data was aggregated into a chronological series containing **305 observed trading-day revenue values**.

The dataset was divided chronologically to preserve the temporal structure required for forecasting.

The resulting series provides the foundation for historical revenue analysis, temporal feature engineering, model development, and rolling-origin evaluation.

**Dataset Source:** [Online Retail Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)

---

## 🎯 Problem Statement

Retail sales revenue fluctuates across trading days due to changing purchasing activity, recurring calendar patterns, seasonal effects, and variations in customer demand.

Understanding these fluctuations can support short-term financial planning, while forecasting future revenue provides a quantitative basis for anticipating periods of higher or lower sales activity.

However, daily revenue forecasting presents several challenges:

- Daily revenue can fluctuate substantially.
- Historical patterns may change over time.
- Calendar effects interact with recent revenue activity.
- High-revenue trading days may be difficult to predict accurately.
- Forecast accuracy can vary across multi-day horizons.
- Statistical and machine learning models may capture different aspects of the revenue series.
- Systematic forecast bias can affect the usefulness of predictions for planning.

The objective of this project is to develop a **7-trading-day time series forecasting framework** that predicts daily net merchandise revenue using historical observations, evaluates multiple forecasting approaches, and provides interpretable insights into forecasting behavior.

---

## 🚀 Project Objectives

- Understand the structure and characteristics of retail transaction data.
- Identify historical revenue patterns, trends, and temporal variations.
- Construct a chronological daily net merchandise revenue series.
- Prepare independent chronological training and testing periods.
- Engineer leakage-aware calendar, lag, rolling, and revenue-change features.
- Develop statistical and machine learning forecasting models.
- Establish naive and seasonal naive forecasting baselines.
- Compare model configurations using chronological validation.
- Optimize selected forecasting configurations.
- Develop and evaluate voting and stacking ensembles.
- Select the final forecasting configuration using validation performance.
- Assess forecasting accuracy using MAE, RMSE, and R².
- Investigate forecast errors across seven trading-day horizons.
- Analyze forecasting performance across different revenue levels.
- Explain model behavior through feature importance and statistical parameter analysis.
- Examine ensemble component predictions and forecasting disagreement.
- Identify practical limitations and opportunities for future forecasting improvements.

---

## 🔍 Exploratory Data Analysis

The EDA stage investigates transaction characteristics and the temporal structure of daily retail revenue.

### Analysis Areas

- Transaction-level dataset characteristics
- Missing values and duplicate observations
- Invoice, quantity, and unit price characteristics
- Transaction revenue distributions
- Daily net merchandise revenue
- Historical revenue trends
- Monthly revenue patterns
- Day-of-week revenue variations
- Recurring trading-cycle patterns
- Revenue variability and fluctuations
- High-revenue and low-revenue trading days
- Temporal relationships relevant to forecasting

### Key Observations

- The original dataset contains **541,909 transaction records**, with **538,998 records** retained after cleaning.
- The cleaned dataset represents total net merchandise revenue of approximately **9.79 million** in the dataset's monetary units.
- Daily aggregation produces **305 observed trading-day revenue observations**.
- Historical revenue exhibits substantial fluctuations, motivating the use of temporal, historical, and seasonal forecasting features.
- Subsequent statistical model analysis identifies significant six-trading-day autoregressive relationships.
- Calendar characteristics and historical rolling revenue statistics provide useful predictive information in the fitted Extra Trees models.

These findings informed the data preparation strategy, feature engineering approach, and selection of forecasting models.

---

## ⚙️ Data Preparation & Feature Engineering

### Data Preparation

The preparation workflow included:

- Inspection and cleaning of transaction-level records.
- Calculation of eligible transaction revenue.
- Aggregation of cleaned transactions into daily net merchandise revenue.
- Chronological ordering of the daily revenue series.
- Preservation of the observed trading-day sequence.
- Creation of independent training and testing periods.
- Saving of processed datasets for reuse.

### Chronological Data Splitting

The daily revenue series was divided chronologically to preserve temporal ordering.

| Component | Date Range | Observations |
|---|---|---:|
| Training | December 1, 2010 – September 29, 2011 | 244 |
| Testing | September 30, 2011 – December 9, 2011 | 61 |
| Total | December 1, 2010 – December 9, 2011 | 305 |

Random splitting was avoided because it would disrupt the temporal structure of the forecasting problem.

### Temporal Feature Engineering

The feature engineering process created **48 predictor features per forecasting horizon** across several feature groups.

#### Calendar Features

Calendar variables capture recurring temporal characteristics associated with forecast origins and target trading dates.

Examples include:

```text
Origin_Month
Origin_Day_Of_Week
Origin_Day_Of_Year
Target_Month
Target_Day_Of_Week
Target_Is_Weekend
```

Target-date features are generated separately for each forecasting horizon.

#### Historical Lag Features

Lag variables represent historical daily revenue observations.

Examples include:

```text
Revenue_Lag_1
Revenue_Lag_3
```

Lagged revenue features help represent short-term historical dependencies and recurring trading-cycle patterns.

#### Rolling Statistics

Rolling features summarize historical revenue levels and variability.

Examples include:

```text
Revenue_Rolling_Min_6
Revenue_Rolling_Min_14
Revenue_Rolling_Max_6
```

These variables provide information about recent revenue conditions.

#### Revenue Change Features

Revenue change indicators describe differences between historical revenue observations.

Example:

```text
Revenue_Change_1
```

These features help represent short-term changes in revenue behavior.

### Multi-Horizon Forecast Targets

Seven target variables were constructed:

```text
Target_H1
Target_H2
Target_H3
Target_H4
Target_H5
Target_H6
Target_H7
```

Each target represents revenue at the corresponding future observed trading-day horizon.

The machine learning approach uses separate horizon-specific models to predict each target.

### Forecasting Leakage Prevention

Historical predictors were constructed using revenue information available at the forecast origin.

The forecasting design also preserves the chronological training boundary and avoids using future target revenue values as predictors.

Known future trading-date characteristics are used as calendar inputs.

During rolling-origin evaluation, newly observed actual revenue becomes available for subsequent forecast origins.

### Final Feature Dataset

| Component | Value |
|---|---:|
| Training forecast origins | 207 |
| Testing forecast origins | 55 |
| Total dataset columns | 87 |
| Predictor features per horizon | 48 |
| Forecast targets | 7 |
| Forecast horizon | 7 trading days |

The engineered datasets were saved for reuse during model development, evaluation, and explainability.

---

## 🤖 Forecasting Approach

The project treats daily revenue forecasting as a **supervised multi-step time series prediction problem** with a seven-trading-day forecast horizon.

Statistical forecasting, machine learning, and ensemble approaches were explored to compare their ability to capture historical revenue patterns.

### Modeling Workflow

```text
Daily Net Merchandise Revenue Series
                |
                v
Chronological Train-Test Split
                |
                v
Leakage-Aware Feature Engineering
                |
                v
Baseline Forecasting Models
                |
                v
Statistical and Machine Learning Models
                |
                v
Chronological Validation
                |
                v
Hyperparameter Optimization
                |
                v
Forecasting Model Comparison
                |
                v
Voting and Stacking Ensembles
                |
                v
Final Model Selection
                |
                v
Independent Rolling-Origin Evaluation
```

### Forecasting Model Families

The project explored multiple forecasting approaches, including:

- **Baseline Forecasting:** Naive and Seasonal Naive.
- **Statistical Forecasting:** AutoReg and SARIMA.
- **Machine Learning:** Regression models, including Extra Trees and other tree-based approaches.
- **Ensemble Forecasting:** Voting and stacking combinations of statistical and machine learning predictions.

A total of **29 model configurations and experiments** were included in the development leaderboard.

### Validation Strategy

The project uses chronological validation to compare forecasting models without randomly mixing past and future observations.

**Mean Absolute Error (MAE)** was used as the primary model-selection metric.

### Validation Model Comparison

| Model | Validation MAE |
|---|---:|
| **Voting Hybrid** | **11,690.95** |
| Seasonal Naive | 13,194.90 |

Voting Hybrid achieved approximately **11.40% lower validation MAE** than the Seasonal Naive baseline.

### Selected Forecasting Model

**Voting Hybrid Ensemble**

The ensemble combines forecasts from AutoReg, SARIMA, and Extra Trees.

| Component | Ensemble Weight |
|---|---:|
| AutoReg | 33.33% |
| SARIMA | 33.33% |
| Extra Trees | 33.33% |
| **Total** | **100%** |

The ensemble prediction is calculated as:

\[
\hat{y}_{ensemble}
=
\frac{
\hat{y}_{AutoReg}
+
\hat{y}_{SARIMA}
+
\hat{y}_{ExtraTrees}
}{3}
\]

The ensemble was selected using validation performance before examining the independent test results.

### Selected Component Configurations

**AutoReg**

```text
Lags: [1, 2, 3, 6, 12]
```

**SARIMA**

```text
Non-seasonal order: (1, 0, 1)
Seasonal order: (1, 0, 1, 6)
```

**Extra Trees**

```text
n_estimators: 100
max_depth: 5
min_samples_leaf: 5
Horizon-specific models: 7
Predictor features per horizon: 48
```

During rolling-origin evaluation, AutoReg and SARIMA were refitted using the historical revenue available at each forecast origin.

The seven Extra Trees models used their fitted training configurations to generate horizon-specific predictions.

---

## 📈 Model Evaluation

The validation-selected Voting Hybrid ensemble and comparison models were evaluated on an independent chronological test period containing **61 trading-day observations**.

### Evaluation Strategy

The evaluation used **55 rolling forecast origins**.

Each origin generated forecasts for the next seven observed trading days.

This produced:

- **55 forecast origins**
- **7 forecast horizons**
- **385 forecast records**

Forecast windows overlap, meaning some actual trading dates appear in multiple records at different forecast horizons.

### Final Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Extra Trees** | **12,374.70** | **16,749.57** | **0.1244** |
| Voting Hybrid | 12,465.39 | 17,080.80 | 0.0894 |
| AutoReg | 13,303.88 | 17,386.78 | 0.0565 |
| Seasonal Naive | 13,940.46 | 18,311.69 | -0.0466 |
| SARIMA | 14,131.27 | 18,703.35 | -0.0918 |
| Naive | 18,349.33 | 23,467.37 | -0.7188 |

### Selected Ensemble Performance

| Metric | Result |
|---|---:|
| Mean Absolute Error | **12,465.39** |
| Root Mean Squared Error | 17,080.80 |
| R² Score | 0.0894 |
| Mean Absolute Percentage Error | 25.35% |
| Improvement over Seasonal Naive MAE | 10.58% |
| Test trading days | 61 |
| Rolling forecast origins | 55 |
| Forecast records | 385 |

### Evaluation Analysis

The evaluation included:

- Comparison of forecasting models on the chronological test period.
- Rolling-origin forecast generation.
- Actual versus predicted revenue analysis.
- Prediction error distributions.
- Forecast-horizon performance analysis.
- Revenue-level performance analysis.
- Forecast bias diagnostics.
- Comparison of validation and test results.

### Horizon-Wise Ensemble Performance

| Forecast Horizon | Voting Hybrid MAE |
|---|---:|
| Horizon 1 | 11,850.90 |
| Horizon 2 | 11,653.01 |
| Horizon 3 | 12,619.77 |
| Horizon 4 | 12,422.31 |
| Horizon 5 | 12,665.20 |
| Horizon 6 | 12,216.19 |
| Horizon 7 | 13,830.32 |

### Revenue-Level Error Analysis

| Revenue Level | Forecast Records | MAE | Underprediction Rate |
|---|---:|---:|---:|
| Low | 135 | 6,132.81 | 42.96% |
| Medium | 127 | 8,283.38 | 81.10% |
| High | 123 | 23,733.78 | 98.37% |

### Key Evaluation Findings

- **Extra Trees** achieved the lowest test MAE of **12,374.70**, narrowly outperforming Voting Hybrid.
- **Voting Hybrid** remained the designated final model because it was selected using validation performance rather than test results.
- The selected ensemble achieved a **10.58% reduction in test MAE** compared with Seasonal Naive.
- Horizon 2 recorded the lowest ensemble MAE, while Horizon 7 recorded the highest.
- High-revenue observations produced substantially larger absolute forecasting errors.
- Voting Hybrid underestimated actual revenue in **73.25%** of forecast records.
- The ensemble's test R² of **0.0894** indicates that substantial revenue variation remains unexplained.

These findings demonstrate that forecasting performance varies across revenue levels and prediction horizons.

---

## 🔎 Model Explainability

Model explainability focused on understanding the behavior of the Voting Hybrid ensemble and identifying influential predictors within its Extra Trees component.

### Explainability Methods

- Extra Trees feature importance.
- Feature importance aggregation across seven horizons.
- Horizon-specific feature importance.
- Permutation importance for Horizon 1.
- Comparison of impurity-based and permutation-based feature rankings.
- AutoReg coefficient analysis.
- SARIMA parameter analysis.
- Statistical model temporal structure comparison.
- Ensemble forecast reconstruction.
- Individual ensemble component performance.
- Component forecast disagreement analysis.
- Horizon-wise ensemble diagnostics.

### Ensemble Forecast Reconstruction

The ensemble predictions were reconstructed using the equal-weight average of AutoReg, SARIMA, and Extra Trees predictions.

The reconstructed forecasts were compared against the saved Voting Hybrid predictions.

The reconstruction passed the numerical consistency check, confirming that the stored ensemble predictions matched the component averaging calculation within the specified tolerance.

### Individual Ensemble Component Performance

| Model | Test MAE | Mean Prediction |
|---|---:|---:|
| **Extra Trees** | **12,374.70** | 39,520.24 |
| Voting Hybrid | 12,465.39 | 40,073.94 |
| AutoReg | 13,303.88 | 42,138.31 |
| SARIMA | 14,131.27 | 38,563.27 |

### Key Component Findings

- Extra Trees achieved the lowest test MAE among the ensemble components.
- AutoReg produced the highest average revenue prediction.
- SARIMA produced the lowest average revenue prediction.
- All three components underestimated mean actual revenue.
- Voting Hybrid did not outperform Extra Trees on the independent test period.

### Ensemble Disagreement Analysis

Component disagreement was measured using the standard deviation of the three component forecasts.

| Metric | Result |
|---|---:|
| Mean component disagreement | 3,645.69 |
| Median component disagreement | 3,093.39 |
| Highest mean disagreement horizon | Horizon 7 |
| Horizon 7 mean disagreement | 4,525.87 |
| Disagreement-error correlation | -0.1396 |

The weak negative correlation indicates that component disagreement is not a reliable standalone indicator of forecasting error.

### Extra Trees Feature Importance

Feature importance was calculated across the seven fitted Extra Trees horizon-specific models.

#### Top 10 Predictors

| Rank | Feature | Mean Importance |
|---:|---|---:|
| 1 | `Target_Is_Weekend` | 35.59% |
| 2 | `Target_Day_Of_Week` | 7.83% |
| 3 | `Origin_Day_Of_Year` | 5.75% |
| 4 | `Origin_Month` | 5.01% |
| 5 | `Target_Month` | 4.15% |
| 6 | `Origin_Day_Of_Week` | 2.58% |
| 7 | `Revenue_Rolling_Min_14` | 2.39% |
| 8 | `Revenue_Rolling_Min_6` | 2.06% |
| 9 | `Revenue_Rolling_Max_6` | 1.55% |
| 10 | `Revenue_Change_1` | 1.52% |

Calendar-related features account for several of the most influential predictors.

`Target_Is_Weekend` is the highest-ranked feature across all seven forecasting horizons.

### Permutation Importance

Permutation importance was evaluated for the Horizon 1 Extra Trees model.

| Feature | Mean MAE Increase |
|---|---:|
| `Target_Is_Weekend_H1` | 3,440.59 |
| `Revenue_Rolling_Min_14` | 829.55 |
| `Target_Day_Of_Week_H1` | 177.95 |

The Horizon 1 baseline MAE was **11,671.25**.

The Spearman correlation between impurity-based and permutation-based feature rankings was **0.4416**, indicating moderate agreement between the two explanation methods.

### Statistical Model Interpretation

The AutoReg model identified statistically significant positive coefficients for:

- Revenue Lag 6: **0.2951**
- Revenue Lag 12: **0.1797**

The SARIMA model identified statistically significant coefficients for:

- AR(1): **-0.6656**
- MA(1): **0.7560**
- Seasonal AR(6): **0.5571**

Both models identified significant six-trading-day autoregressive relationships.

### Explainability Findings

- Calendar characteristics are highly influential within the Extra Trees forecasting models.
- Historical rolling revenue statistics provide additional predictive information.
- Six-trading-day seasonal relationships are supported by both AutoReg and SARIMA parameter analysis.
- Impurity-based and permutation-based importance methods identify some common influential features but differ in their broader rankings.
- Ensemble component predictions exhibit meaningful differences across forecasting horizons.
- The selected ensemble demonstrates persistent revenue underprediction.

**Interpretation Note:** Feature importance measures predictive importance within the fitted Extra Trees models. Statistical coefficients describe modeled relationships, not causal effects. These explanations do not directly establish causal drivers of revenue or explain every aspect of the complete Voting Hybrid ensemble.

---

## 💡 Key Findings

- The original retail dataset was transformed into a chronological daily revenue forecasting series containing **305 observed trading days**.
- Cleaned transaction records represent approximately **9.79 million** in net merchandise revenue.
- Calendar, lag, rolling, and revenue-change features provided a structured representation of historical revenue behavior.
- Chronological validation supported forecasting model comparison without randomly mixing past and future observations.
- **Voting Hybrid** achieved the lowest selected validation MAE of **11,690.95**.
- **Extra Trees** achieved the lowest test MAE of **12,374.70**.
- The selected ensemble achieved a test MAE of **12,465.39**, improving upon Seasonal Naive by **10.58%**.
- Forecasting errors varied across the seven trading-day horizons.
- High-revenue observations recorded substantially larger absolute forecasting errors.
- The ensemble underestimated actual revenue in **73.25%** of test forecast records.
- `Target_Is_Weekend` accounted for approximately **35.59%** of mean Extra Trees feature importance.
- AutoReg and SARIMA both identified statistically significant six-trading-day autoregressive relationships.
- The explainability analysis demonstrated the relevance of calendar characteristics, historical revenue patterns, and ensemble component differences.
- The project illustrates both the potential and limitations of short-term retail revenue forecasting using historical transaction data.

---

## ⚠️ Limitations & Future Improvements

### Current Limitations

- The dataset covers a relatively limited historical period from December 2010 to December 2011.
- The daily revenue series contains only 305 observed trading days, restricting the amount of historical information available for modeling.
- The forecasting horizon represents observed trading days rather than consecutive calendar days.
- The forecasting framework assumes future trading dates are known.
- The models do not explicitly incorporate all external revenue drivers, such as promotions, holidays, marketing campaigns, and broader economic conditions.
- The selected ensemble achieved a test R² of 0.0894, indicating substantial unexplained revenue variation.
- Forecast errors remain relatively high during high-revenue trading periods.
- The selected ensemble exhibits systematic underprediction.
- Rolling-origin evaluation generates overlapping forecast records, so the 385 evaluation records are not independent observations.
- Model performance was evaluated on a limited historical holdout period.
- Equal ensemble weights do not guarantee that the ensemble outperforms its individual components.
- Tree-based feature importance can be influenced by correlations among predictors.
- Statistical significance does not establish causal relationships or guarantee forecasting accuracy.
- Generalization to other retailers, markets, and future time periods remains unverified.

### Future Improvements

- Extend the dataset to include multiple years of historical revenue observations.
- Incorporate holiday calendars, promotional events, and marketing information.
- Investigate additional seasonal and trend modeling approaches.
- Explore improved feature representations for unusually high-revenue trading days.
- Evaluate alternative loss functions and bias-correction strategies.
- Investigate horizon-specific ensemble weighting using validation data.
- Develop prediction intervals or probabilistic forecasts to represent uncertainty.
- Assess model stability through additional chronological backtesting.
- Explore alternative feature importance and model explanation techniques.
- Develop a production-oriented forecasting pipeline with automated monitoring.
- Implement periodic model retraining and performance reassessment.
- Validate forecasting performance on newer retail transaction data.

---

## 📁 Notebook Workflow

The project is organized into seven notebooks.

| Notebook | Purpose |
|---|---|
| `01_data_understanding.ipynb` | Dataset inspection, transaction characteristics, and initial quality assessment |
| `02_exploratory_data_analysis.ipynb` | Retail revenue patterns and temporal exploratory analysis |
| `03_data_preparation.ipynb` | Transaction cleaning, daily aggregation, and chronological splitting |
| `04_feature_engineering.ipynb` | Calendar, lag, rolling, and multi-horizon feature creation |
| `05_model_development.ipynb` | Forecasting model development, tuning, validation, and ensemble selection |
| `06_model_evaluation.ipynb` | Independent rolling-origin evaluation and forecasting error diagnostics |
| `07_model_explainability.ipynb` | Feature importance, statistical model interpretation, and ensemble analysis |

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/TejasMJ/Retail-Sales-Dynamics.git
cd Retail-Sales-Dynamics
```

*Update the repository URL if your published GitHub repository uses a different name.*

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

#### Windows

```bash
.\.venv\Scripts\activate
```

#### Mac/Linux

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Prepare the Dataset

Download the original Online Retail dataset from the UCI Machine Learning Repository.

Place the downloaded dataset in the `data/raw/` directory using the filename expected by the data-understanding notebook.

### 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the `notebooks/` directory and execute the notebooks in numerical order.

### 7. Execute the Forecasting Workflow

Follow the seven notebooks sequentially:

1. Data Understanding
2. Exploratory Data Analysis
3. Data Preparation
4. Feature Engineering
5. Model Development
6. Model Evaluation
7. Model Explainability

**Note:** Model development includes repeated chronological validation, statistical model fitting, machine learning experimentation, and ensemble evaluation. Execution times may vary depending on available computing resources.

The evaluation and explainability notebooks depend on processed datasets and model artifacts generated during the preceding stages.

---

## 📌 Project Status

**Status:** Technical Implementation Completed ✅

The project includes:

- Data understanding and transaction cleaning.
- Exploratory Data Analysis.
- Daily revenue time series preparation.
- Chronological train-test splitting.
- Temporal and historical feature engineering.
- Multi-horizon forecasting target creation.
- Baseline forecasting experimentation.
- AutoReg and SARIMA statistical forecasting.
- Machine learning model development.
- Chronological model validation.
- Hyperparameter optimization.
- Voting and stacking ensemble experimentation.
- MLflow experiment tracking.
- Final model selection using validation performance.
- Independent rolling-origin evaluation.
- Seven-horizon performance assessment.
- Revenue-level error analysis.
- Forecast bias diagnostics.
- Extra Trees feature importance.
- Permutation importance.
- Statistical model parameter interpretation.
- Ensemble forecast reconstruction.
- Component disagreement analysis.
- Reusable artifact preservation.

**Final Validation-Selected Model:** Voting Hybrid Ensemble.

**Validation MAE:** 11,690.95.

**Test MAE:** 12,465.39.

**Test RMSE:** 17,080.80.

**Test R²:** 0.0894.

**Test MAPE:** 25.35%.

**Improvement Over Seasonal Naive:** 10.58% lower test MAE.

**Lowest Test MAE:** Extra Trees — 12,374.70.

---

## 👨‍💻 Author

**Tejas Jadhav**

- GitHub: [@tejas-jadhav](https://github.com/TejasMJ)
- LinkedIn: [Tejas Jadhav](https://www.linkedin.com/in/tejas-m-jadhav/)
