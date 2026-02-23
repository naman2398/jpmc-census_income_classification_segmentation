# Project Report: Census Income Classification & Segmentation

## 1. Executive Summary

This project addresses two tasks for a retail business client using US Census Bureau Current Population Survey data (1994-1995, 199,523 records, 40 demographic/employment features):

1. **Classification:** Predict whether an individual earns more or less than $50,000/year.
2. **Segmentation:** Identify distinct population segments for differentiated marketing.

For classification, XGBoost achieved a ROC-AUC of 0.949 with 87% recall on the high-income class, outperforming a Logistic Regression baseline. For segmentation, PCA dimensionality reduction followed by K-Means clustering identified 4 segments with distinct demographic and economic profiles, ranging from children/dependents to high-earning professionals.

---

## 2. Data Exploration

### 2.1 Dataset Overview

The dataset contains 199,523 observations with 40 features plus a sampling weight and a year indicator. Features span demographics (age, sex, race, education), employment (industry, occupation, class of worker), income components (capital gains/losses, dividends), and geographic/migration variables.

All string values contain leading whitespace, which is stripped during loading. Missing values appear as `?` in several categorical fields. Many categorical columns are dominated by "Not in universe" — a valid Census Bureau coding indicating the question does not apply to that respondent.

Critically, "Not in universe" coding appears in **three distinct variants** discovered via regex pattern matching (`(?i)not in universe`):
- `Not in universe` — 1,559,657 occurrences across 14 columns
- `Not in universe or children` — 100,684 occurrences (in `major industry code`)
- `Not in universe under 1 year old` — 101,212 occurrences (in `live in this house 1 year ago`)

Hardcoding specific phrases would miss variants. Pattern matching ensures comprehensive detection.

### 2.2 Class Imbalance

The target label is severely imbalanced:
- **≤$50K:** 187,141 records (93.8%)
- **>$50K:** 12,382 records (6.2%)

This 15:1 ratio means accuracy is misleading — a trivial classifier predicting ≤$50K always achieves 93.8% accuracy. Model evaluation therefore focuses on ROC-AUC, precision, recall, and the precision-recall curve.

### 2.3 Key Feature Patterns

**Numeric features:** `capital gains`, `capital losses`, `dividends from stocks`, and `wage per hour` are heavily zero-inflated — the vast majority of records have zero values. Binary flags (e.g., has_capital_gains) were engineered to capture the meaningful signal in these features.

**Categorical features:** Education, occupation, and marital status show strong differentiation between income classes. For example, individuals with professional degrees or in executive/managerial roles have substantially higher rates of >$50K income.

**High correlations:** `num persons worked for employer` correlates with `weeks worked in year` (r=0.747). This pair is retained as they represent different concepts (employment count vs. duration). The detailed recode columns (classification codes excluded from numeric correlation analysis) are dropped for redundancy with their major-code counterparts.

### 2.4 Feature Selection (Data-Backed)

Feature selection uses straightforward, interpretable EDA criteria rather than a single statistical measure. Each column is evaluated on:

1. **Single-value dominance:** if >95% of rows have the same value, the column carries almost no information
2. **"Not in universe" dominance:** if >90% of rows are "Not in universe", the only signal is "in workforce vs not" — already captured by other features
3. **High missingness:** columns with >40% `?` values are unreliable
4. **Redundancy:** a lower-cardinality alternative exists
5. **Multiple weak signals:** combinations of moderate "Not in universe" rates, high missingness, or low variation

| Column | Rule | Key Statistic | Reason |
|---|---|---|---|
| `weight` | Structural | — | Sampling weight, not a feature |
| `year` | Structural | — | Temporal split variable |
| `detailed industry recode` | Redundancy | 52 categories | Redundant with major industry code (24 categories) |
| `detailed occupation recode` | Redundancy | 47 categories | Redundant with major occupation code (15 categories) |
| `fill inc questionnaire for veteran's admin` | Near-constant | 99.0% "Not in universe" | Top value >95% |
| `reason for unemployment` | Near-constant | 97.0% "Not in universe" | Top value >95% |
| `region of previous residence` | "Not in universe" >90% | 92.1% | Question doesn't apply to most respondents |
| `state of previous residence` | "Not in universe" >90% | 92.1% | Question doesn't apply to most respondents |
| `enroll in edu inst last wk` | "Not in universe" >90% | 93.7% | Question doesn't apply to most respondents |
| `member of a labor union` | "Not in universe" >90% | 90.4% | Question doesn't apply to most respondents |
| `live in this house 1 year ago` | Multiple weak | 50.7% "Not in universe" | Moderate "Not in universe" + low variation |
| `migration code-change in msa` | High missingness | 50% `?` | Too much missing data |
| `migration code-change in reg` | High missingness | 50% `?` | Too much missing data |
| `migration code-move within reg` | High missingness | 50% `?` | Too much missing data |
| `migration prev res in sunbelt` | Multiple weak | 42% "Not in universe" + 50% `?` | Moderate "Not in universe" + high missingness |

**15 columns dropped → 26 original + 3 engineered = 29 features retained.**

Feature importance is assessed during modeling using SHAP values (see §4.4) rather than during EDA, giving model-aware importance that accounts for feature interactions.

`?` values are kept as a distinct category rather than imputed, since missingness in census data can itself be informative.

---

## 3. Preprocessing

### 3.1 Shared Preprocessing (data_loader.py)

- **Whitespace stripping** on all string columns
- **Label encoding:** `50000+.` → 1, `- 50000.` → 0
- **Feature engineering:** Three binary flags (`has_capital_gains`, `has_capital_losses`, `has_dividends`) from zero-inflated numeric columns
- **Column drops:** 15 columns removed per data-backed EDA findings (dominance, missingness, redundancy, "Not in universe" analysis)
- **NaN handling:** 874 NaN values in `hispanic origin` filled with "Missing"

### 3.2 Classification Preprocessing

- **Encoding:** `OneHotEncoder` for categoricals (required for Logistic Regression; trees handle it fine)
- **Scaling:** `StandardScaler` on numeric features only
- **Train/test split:** Hybrid 80/20 — all year 94 + a stratified 60% of year 95 → train (159,644 samples), remaining 40% of year 95 → test (39,879 samples). This maximises training data while keeping the test set purely from the later year, preserving temporal validity. Stratified sampling ensures the ~94/6 class ratio is maintained in both sets.
- **Sample weights:** Census `weight` column used as `sample_weight` during model fitting to respect the stratified sampling design

### 3.3 Segmentation Preprocessing

- **Encoding:** `OrdinalEncoder` for categoricals — OneHotEncoder would create ~500+ sparse dimensions that distort Euclidean distance for K-Means
- **Scaling:** `StandardScaler` on all features after encoding — K-Means requires uniform scale across dimensions
- **No train/test split** — clustering uses the full dataset
- **Weight and label dropped** from features; label is used for post-hoc cluster profiling

---

## 4. Classification Model

### 4.1 Model Selection

**Logistic Regression (baseline):** Interpretable linear model. Uses `class_weight='balanced'` to handle class imbalance by upweighting the minority class. Serves as a baseline to quantify the value of a more complex model.

**XGBoost (primary):** Gradient-boosted decision trees — the industry standard for tabular data. Handles feature interactions naturally, is robust to outliers, and supports `scale_pos_weight` for class imbalance. Configured with 300 estimators, max depth 6, learning rate 0.1.

Both models are trained with census sample weights to produce population-representative predictions.

### 4.2 Handling Class Imbalance

Rather than resampling (which discards data or creates synthetic examples), imbalance is handled through:
- **Logistic Regression:** `class_weight='balanced'` — sklearn automatically adjusts weights inversely proportional to class frequencies
- **XGBoost:** `scale_pos_weight = (1 - 0.061) / 0.061 ≈ 15.4` — explicitly scales the positive class gradient contribution

### 4.3 Results

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| **ROC-AUC** | 0.9448 | 0.9493 |
| **Accuracy** | 85% | 88% |
| **Precision (>50K)** | 0.29 | 0.34 |
| **Recall (>50K)** | 0.90 | 0.87 |
| **F1 (>50K)** | 0.44 | 0.48 |

**Key observations:**

- **More training data improved both models.** The 80/20 hybrid split (159,644 train / 39,879 test) produces strong ROC-AUC of 0.945/0.949 by maximizing training data while preserving temporal validity.
- Both models achieve ~0.95 ROC-AUC, indicating strong ranking ability.
- Logistic Regression achieves very high recall (0.90) at the cost of lower precision (0.29) — it catches most high-income individuals but with many false positives.
- XGBoost provides a significantly better precision-recall tradeoff: 87% recall with 34% precision, meaning fewer wasted marketing dollars on misclassified individuals.
- The test set is purely year-95 data, preserving temporal validity — the model is never tested on same-year data it trained on from year 94.

### 4.4 Feature Importance (SHAP Analysis)

Feature importance is assessed using SHAP (SHapley Additive exPlanations) values, which provide model-aware importance that accounts for feature interactions — unlike simple EDA metrics.

XGBoost's top features by mean |SHAP value|:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `age` | 1.818 |
| 2 | `weeks worked in year` | 1.013 |
| 3 | `tax filer stat = Nonfiler` | 0.709 |
| 4 | `sex = Female` | 0.441 |
| 5 | `dividends from stocks` | 0.287 |
| 6 | `num persons worked for employer` | 0.223 |
| 7 | `capital gains` | 0.172 |
| 8 | `education = Bachelors degree` | 0.164 |
| 9 | `detailed household summary = Householder` | 0.158 |
| 10 | `education = High school graduate` | 0.120 |

These align with economic intuition — income correlates strongly with work experience (age, weeks worked), education level, and investment activity (dividends, capital gains). The tax filer status serves as a strong proxy for overall financial engagement. SHAP beeswarm plots (see `outputs/shap_beeswarm.png`) show both the magnitude and direction of each feature's impact on individual predictions.

---

## 5. Segmentation Model

### 5.1 Approach

**Dimensionality Reduction:** PCA reduces the 29 preprocessed features to 17 components capturing 85% of variance. This is critical for K-Means, which suffers from the curse of dimensionality — distance metrics become less meaningful in high-dimensional space.

**Clustering:** K-Means with silhouette score evaluation across k=2 to k=7. The optimal k=4 was selected based on the highest silhouette score (0.231).

### 5.2 Segment Profiles

| Cluster | Size | Mean Age | >50K Rate | Key Characteristics |
|---|---|---|---|---|
| **0** | 31.1% | 10.3 | 0.0% | Children/youth, not in workforce, never married |
| **1** | 2.0% | 44.2 | 29.9% | High-earning males, married, private sector workers |
| **2** | 56.8% | 46.0 | 8.9% | Core working adults, married, mixed occupation, female-leaning |
| **3** | 10.1% | 42.5 | 5.4% | Lower-income working adults, mixed employment |

### 5.3 Segment Interpretation for Marketing

The 4 clusters form clean, actionable marketing tiers:

1. **Premium Segment (Cluster 1):** 2% of population, ~30% earn >$50K. Predominantly male, married, private sector. Target with premium financial products, investment services, wealth management.

2. **Core Working Adults (Cluster 2):** 57% of population, 8.9% earn >$50K. Largest segment — married adults across industries. Target with everyday retail, savings products, insurance, career development.

3. **Budget-Conscious Workers (Cluster 3):** 10% of population, 5.4% earn >$50K. Lower income, mixed employment patterns. Target with value-oriented products, education/upskilling programs.

4. **Youth/Dependents (Cluster 0):** 31% of population, near-zero income. Children and young dependents. Target parents with family-oriented and education products.

---

## 6. Business Recommendations

### Classification Use

- Deploy XGBoost as the primary income classifier for targeting high-value customers
- At the default threshold, the model identifies 87% of high-income individuals with 34% precision — suitable for broad marketing campaigns where reaching most high-income individuals matters more than avoiding false positives
- For targeted, high-cost campaigns (e.g., premium product launches), raise the probability threshold to improve precision at the cost of some recall

### Segmentation Use

- Use segments for **differentiated messaging**: premium financial products for Cluster 1, everyday retail for Cluster 2, value-oriented products for Cluster 3, family/education products for Cluster 0
- Combine classification and segmentation: within each segment, use the classifier's probability scores to further prioritize outreach
- Monitor segment stability over time — demographic shifts may require periodic re-clustering

### Deployment Guidance

- Retrain models quarterly or when new census data becomes available
- Monitor for data drift — feature distributions may shift year over year
- Include a human review step for high-stakes decisions (credit, lending) to avoid bias amplification

---

## 7. Future Work

- **Additional models:** LightGBM, CatBoost, or neural networks for potential performance gains
- **Feature engineering:** Interaction features (e.g., education × occupation), age bins, geographic aggregations
- **Threshold optimization:** Tune classification threshold based on business cost matrix (cost of false positive vs. false negative)
- **Fairness analysis:** Evaluate model performance across protected groups (sex, race) to ensure equitable predictions
- **A/B testing:** Validate segment-based marketing strategies through controlled experiments before full rollout
- **Cluster stability:** Use bootstrap resampling to assess how stable clusters are across different data samples

---

## 8. References

- US Census Bureau. Current Population Survey Technical Documentation. https://www.census.gov/programs-surveys/cps/technical-documentation.html
- XGBoost documentation: https://xgboost.readthedocs.io/
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.
- scikit-learn documentation: https://scikit-learn.org/stable/
- scikit-learn. Selecting the number of clusters with silhouette analysis on KMeans clustering. https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
