<table border="0">
 <tr>
    <td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="150" alt="University Logo" /></td>
    <td>
      <p>University of Pristina</p>
      <p>Faculty of Electrical and Computer Engineering</p>
      <p>Computer and Software Engineering - Master Program</p>
	<p>Course: Machine Learning</p>
      <p>Professor: Dr. Lule Ahmedi</p>
      <p>Assistant: Dr. Sc. Mërgim Hoti</p>
    </td>
 </tr>
</table>

# Property Buyer Prediction

## Project Overview

This repository contains a **production-grade data engineering and machine learning pipeline** designed to predict the most likely buyer of asset sales using Random Forest classification. The project demonstrates advanced techniques in **data cleaning, domain logic application, feature engineering, and dataset preparation** for supervised learning tasks.

The pipeline transforms a real-world, messy business dataset into a clean, feature-rich, ML-ready dataset through systematic application of domain knowledge, statistical methods, and data engineering best practices. The final output is a well-curated dataset with zero missing values, engineered features, and explicit handling of data quality issues.

---
## Data Sources

1. **Agency of Privatization of Kosovo (PAK)**
   - Real estate sales data: prices, locations, property types, buyers
   - Website: https://www.pak-ks.org/page.aspx?id=1,37

2. **Agjencia e Regjistrimit te Bizneseve (ARBK)**
   - Business information linked to buyer names
   - Enriches the dataset with additional business context
   - Website: https://arbk.rks-gov.net/
---

## Objective

### Problem Statement
Asset sales in emerging markets often involve complex business relationships and occur through various transaction types. Understanding which buyer (e.g., individual, business entity, investor, etc.) is most likely to purchase a given asset based on its characteristics, location, and market conditions provides significant value for:

- **Market analysis** – identifying buyer patterns and market segmentation
- **Pricing optimization** – understanding buyer-specific value drivers
- **Risk assessment** – identifying unusual or high-risk transactions
- **Business strategy** – targeting the right buyer segments for specific assets

### Target Variable
The model predicts **bleresi** (buyer) – a categorical variable representing the buyer classification at the time of asset sale.

### Application Domain
The dataset combines:
- **Asset characteristics** – property type (land, building/object, mixed assets), location, area
- **Transaction details** – sale price, contract date
- **Business context** – buyer business registration information (ARBK – Albanian business registry data)

---
# Phase 1 : Model Preperation
## Raw Dataset Description

### Partial Datasets
- `./data/raw/sales.xlsx`
- `./data/raw/rent.xlsx`
- `./data/raw/arbk_2025-03-10.txt` (after extracting)

### Translated Datasets
In order to have more rows to work with, we got the rent.xlsx and applied some transformations to it to show the values of the rent as how it would have been if it were a sale so `rent.xlsx` -> `sales_with_converted_rents.xlsx`

### Preprocessed Dataset
After adding the rows of `sales_with_converted_rents.xlsx` to `./data/raw/sales.xlsx` and then combining them we would have a final dataset to be our initial start
**File**: `sales_with_converted_rents_enriched_with_arbk.xlsx` which we will be refering to as "the" or "the original" dataset

### Content Overview
The raw dataset encompasses real-world asset sale transactions enriched with business registry information from ARBK (Albanian business registration system). Each row represents a single asset sale transaction.

### Original Dataset Schema (High-Level)
| Category | Examples |
|----------|----------|
| **Asset Information** | Category, Type, Location (Municipality), Address, Area (object/land) |
| **Transaction Details** | Sale Price, Contract Date, Sale Method |
| **Business Context** | Company Name, Business Type, Capital, Number of Employees, Registration Date |
| **Enrichment** | ARBK matching ID, Business activities, Business registration status |

### Dataset Characteristics
- **Language**: `Albanian` (original column names and categorical values)
- **Size**: `3,483` transactions (varies by version)
- **Time Period**: Multi-year asset sales history
- **Features**: Mixed types (text, numeric, date, categorical)
## Dataset Key Findings

### Dataset Overview
- **Total Records**: 3,483 property sales
- **Total Features**: 47 columns (numeric, categorical, date/time)
- **Data Types**: Mix of integers, floats, objects, and datetime columns

### Data Quality
**No Duplicate Rows**: 100% unique records (0 duplicates)


### Data Characteristics
- **Missing Values**: Present in several columns (see notebook for detailed analysis)
- **Categorical Cardinality**: Analyzed to identify low-cardinality columns (<= 20 unique values) suitable for feature engineering
- **Visualizations**: Data types distribution, null values heatmap, ARBK match rate pie chart, cardinality bar charts

*See `notebooks/01_data_exploration.ipynb` for complete tables, visualizations, and detailed analysis.*

---

## Initial Problems in the Raw Dataset

The raw dataset contained significant data quality challenges that prevented immediate machine learning application:

### 1. **Column Naming Issues**
- Column names in Albanian with special characters: e.g., "Sipërfaqja e tokës ne metra katror"
- Inconsistent naming conventions (mixed spaces, special characters, case variations)
- **Impact**: Difficult to programmatically process and merge with other systems

### 2. **Numeric Formatting Problems**
- Currency symbols mixed with values: `€12,500.50`
- Percentage symbols: `85%`
- Whitespace padding: ` 1000 `
- Comma decimal separators in some fields
- Mixed text and numeric content in same column
- **Impact**: Cannot perform numeric calculations or statistical analysis

### 3. **Invalid and Missing Data**
- Null business type (ARBK_nLlojiBiznesitID): ~15-20% of records
- Null target variable (blerësi): ~8-12% of records
- Missing area values for both land and object
- Missing business capital or employee counts
- **Impact**: Cannot train supervised learning model on incomplete data

### 4. **Logical Inconsistencies**
- Land-only assets (no building) with object area values > 0
- Object-only assets (building only) with land area values > 0
- Inconsistent classification of asset types vs. recorded areas
- **Impact**: Model learns spurious patterns instead of true business relationships

### 5. **Lack of Engineered Features**
- Raw data contains only primitive features
- No derived metrics (price per area, business age, asset complexity)
- Missing temporal features (transaction seasonal patterns)
- Missing ratio/interaction features capturing business dynamics
- **Impact**: Limited model expressiveness and predictive power

### 6. **Outliers and Data Quality Anomalies**
- Extreme price values (potentially data entry errors or genuine outliers)
- Unrealistic area measurements (0 or extremely large)
- Sparse categorical variables (rare buyer or business types)
- **Impact**: Model may overfit to outliers or learn from very few examples per category

---

## Data Preparation Pipeline

The pipeline systematically addresses each data quality challenge through a **4-phase ETL (Extract, Transform, Load) architecture** with 8 specialized transformation components.

### 1: Extract

#### Data Loading
```
Raw Excel File → Read All Rows → Column Name Normalization → In-Memory DataFrame
```

**Process**:
1. Load complete Excel workbook into pandas DataFrame
2. Apply column name normalization function to all column names
3. Log extraction statistics (row/column counts, sample normalizations)

**Output**: Clean, normalized DataFrame ready for transformation

---

### 2: Transform

#### 2.1 Data Validation & Row Filtering

**Mandatory Column Validation**
- Verify `arbk_nllojibiznesitid` (business type) exists after normalization
- Verify `bleresi` (target) column exists
- Raise explicit error if either is missing

**Remove Invalid Records**
```
Remove rows where arbk_nllojibiznesitid IS NULL
  Reason: Cannot classify business context without business type
  Impact: ~15-20% of raw records eliminated
  
Remove rows where bleresi IS NULL
  Reason: Cannot train supervised model without target variable
  Impact: ~8-12% of raw records eliminated
```

**Outcome**: Dataset contains only rows with valid business context and target labels

---

#### 2.2 Column Name Normalization

**Transformation Logic**
```
Input: "Sipërfaqja e tokës ne metra katror"
Process:
  1. Remove/replace Albanian diacritics: ë→e, ç→c, etc.
  2. Remove special characters: parentheses, slashes, colons, quotes
  3. Replace spaces with underscores
  4. Remove all remaining non-alphanumeric characters
  5. Collapse multiple underscores
  6. Convert to lowercase
Output: "siperfaqja_e_tokes_ne_metra_katror"
```

**Scope**: Applied to all 50+ column names
**Benefit**: Enables consistent, programmatic column access; ensures compatibility with ML pipelines

---

#### 2.3 Duplicate Detection and Removal

**Full-Row Duplicates**
```
Identify: Rows with identical values across ALL columns
Remove: Keep first occurrence, remove all subsequent duplicates
Report: Number of full duplicates removed
```

**Business-Key Duplicates**
```
Identify: Rows duplicate on business-relevant key columns:
  - pak_id (asset ID)
  - emri_i_ndermarrjes_se_re_apo_asetit_ne_likuidim (asset name)
  - komuna_lokacioni_i_asetit_te_shitur (municipality)
  - cmimi_i_shitjes_se_asetit (sale price)
  - data_e_kontrates (contract date)
  - bleresi (buyer)
  
Remove: Keep first occurrence (by order in dataset)
Reason: Likely data entry errors or system redundancies
```

**Outcome**: Dataset contains only unique transactions

---

#### 2.4 Numeric Type Conversion

**Targeted Numeric Columns** (45+ columns):
- `cmimi_i_shitjes_se_asetit` (sale price)
- `siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2` (object area)
- `siperfaqja_e_tokes_ne_metra_katror` (land area)
- `arbk_kapitali` (business capital)
- `arbk_numpunetoreve` (number of employees)
- Business activity codes (5 activity fields)
- And others

**Cleaning Algorithm**:
```python
for column in numeric_columns:
    s = column.astype(str).str.strip()
    # Remove empty placeholders
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    # Remove currency symbols
    s = s.str.replace("€", "")
    # Remove percentage symbols
    s = s.str.replace("%", "")
    # Remove thousand separators (commas)
    s = s.str.replace(",", "")
    # Remove all remaining non-numeric characters except decimal point and minus sign
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    # Convert to numeric, coercing errors to NaN
    column = pd.to_numeric(s, errors="coerce")
```

**Result**: All numeric columns now contain clean float values ready for computation

---

#### 2.5 Date Type Conversion

**Targeted Date Columns**:
- `data_e_kontrates` (contract date)
- `arbk_dataregjistrimit` (business registration date)

**Process**:
```
Attempt pd.to_datetime() with error coercion
  → Handles multiple date formats automatically
  → Converts errors to NaT (Not a Time)
  → Creates datetime64 dtype for temporal feature engineering
```

**Benefit**: Enables calculation of temporal features (year, month, quarter, age)

---

#### 2.6 Domain Logic: Asset Classification & Area Inference

This section applies business domain knowledge to eliminate logical inconsistencies.

##### Asset Classification Algorithm

**Text-Based Asset Recognition**
Combines information from multiple asset-related columns to classify each record as one of three types:

```
Classification Rules:

1. LAND_ONLY (Land-Only Asset)
   Pattern: Contains land tokens (tokë, truall, parcelë)
            AND NO object tokens
            AND NOT explicitly mixed
   Examples: "tokë bujqësore", "truall", "parcelë"
   
2. OBJECT_ONLY (Building/Structure Only)
   Pattern: Contains object tokens (ndërtesë, apartament, lokal, fabrikë)
            AND NO land tokens
            AND NOT explicitly mixed
   Examples: "ndërtesë administrative", "apart hotel", "fabrikë"
   
3. MIXED_ASSET (Combined Land and Structure)
   Pattern: Explicitly marked with mixed indicators
   OR both object tokens AND land tokens present
   Examples: "depo dhe tokë", "ndërtesë administrative dhe tokë"
```

##### Business Rules for Area Imputation

Based on asset classification, apply deterministic rules to resolve area inconsistencies:

```
Rule 1: If LAND_ONLY and object_area IS NOT NULL → Set object_area = 0
  Reasoning: Land-only assets cannot have building area
  Impact: Eliminates logical contradiction
  
Rule 2: If OBJECT_ONLY and land_area IS NOT NULL → Set land_area = 0
  Reasoning: Object-only assets are structures without land
  Impact: Eliminates logical contradiction
  
Rule 3: If MIXED_ASSET and either area IS NULL
  Strategy: Use group-based median imputation (see section 2.8)
  Groups: Municipality × Asset Category × Business Type
  Requirement: Minimum 5 observations per group
  Fallback: Use 0 if insufficient group size
```

**Outcome**: 
- Land and object area values now logically consistent with asset type
- No contradictions (e.g., land-only assets with object area)
- Prepared for feature engineering and statistical modeling

---

#### 2.7 Feature Engineering

This section creates 15+ derived features from raw data, capturing domain-specific insights for ML.

##### A. Area-Based Features

**total_area_m2** (Total Property Area)
```
Formula: total_area_m2 = object_area_m2 + land_area_m2
Type: Numeric (continuous)
Meaning: Combined usable/owned area in square meters
ML Value: Captures overall property scale; correlates with price and buyer type
```

**object_to_land_ratio** (Building-to-Land Ratio)
```
Formula: object_area / land_area  (or 0 if land_area = 0)
Type: Numeric (continuous, range [0, ∞))
Meaning: How much of the property is built vs. undeveloped
ML Value: Distinguishes between land development potential and existing structures
Examples:
  – Pure land: ratio = 0
  – Modest building on large land: ratio = 0.1
  – Main building with small land: ratio = 10.0
```

**is_land_only** (Binary: Land-Only Asset)
```
Formula: = 1 if asset_classification = "land_only", else 0
Type: Binary flag
Meaning: Explicit indicator of land-only property
ML Value: Direct encoding of asset type; enables buyer segmentation
```

**has_object** (Binary: Has Building/Structure)
```
Formula: = 1 if asset_classification ∈ {"object_only", "mixed_asset"}, else 0
Type: Binary flag
Meaning: Whether property includes a built structure
ML Value: Separates pure land from properties with improvements
```

**is_large_land** (Binary: Land Area Above Median)
```
Formula: = 1 if land_area_m2 > median(land_area_m2), else 0
Type: Binary flag
Meaning: Land size relative to population median
ML Value: Captures scale-based preferences (investors in large parcels vs. small plots)
```

**asset_structure_type** (Categorical: Asset Structure)
```
Values: {"land_only", "object_only", "mixed_asset", "unknown_asset_type"}
Meaning: Summary asset classification
ML Value: Direct encoding of property structure; enables category-specific modeling
Usage: Can be one-hot encoded into 4 binary features in preprocessing
```

**area_data_completeness_score** (Numeric: Data Quality Flag)
```
Formula: Sum of: has_object_area + has_land_area  (range [0, 2])
Meaning: 0 = both missing, 1 = one area present, 2 = both present
ML Value: Data quality indicator; flags problematic records where imputation occurred
```

##### B. Price-Based Features

**price_per_total_m2** (Price Per Total Square Meter)
```
Formula: sale_price / total_area_m2  (or 0 if total_area = 0)
Type: Numeric (continuous, units: USD/m² or local currency per m²)
Meaning: Price intensity per unit area
ML Value: Normalizes price for property size; enables cross-property comparison
Examples:
  – Prime urban location: $500-1000/m²
  – Rural land: $10-50/m²
  – Mixed commercial: $200-600/m²
```

**price_per_land_m2** (Price Per Land Square Meter)
```
Formula: sale_price / land_area_m2  (or 0 if land_area = 0)
Type: Numeric (continuous)
Meaning: Land value component per unit
ML Value: Separates land value from building value; indicates location premium
```

**price_per_object_m2** (Price Per Building Square Meter)
```
Formula: sale_price / object_area_m2  (or 0 if object_area = 0)
Type: Numeric (continuous)
Meaning: Building/structure value per unit area
ML Value: Captures construction quality and functional utility value
```

**log_sale_price** (Log-Transformed Sale Price)
```
Formula: log_sale_price = log(1 + sale_price)
Type: Numeric (continuous)
Meaning: Log-normalized price for better statistical distribution
ML Value: Stabilizes variance for linear models; better representation of multiplicative relationships
```

**is_high_value_asset** (Binary: Price Above Median)
```
Formula: = 1 if sale_price > median(sale_price), else 0
Type: Binary flag
Meaning: Asset priced above market median
ML Value: Segments premium/luxury transactions; may have distinct buyer patterns
```

**area_price_interaction** (Interaction Feature)
```
Formula: total_area_m2 × price_per_total_m2
Type: Numeric (continuous)
Meaning: Captures non-linear interaction between size and price intensity
ML Value: Exposes compound relationships (e.g., large properties with high unit price = premium assets)
```

##### C. Business Relationship Features

**capital_to_sale_price_ratio** (Buyer Capital to Transaction Value)
```
Formula: buyer_business_capital / sale_price  (or 0 if sale_price = 0)
Type: Numeric (continuous, range [0, 1+])
Meaning: Proportion of buyer business capital representing this transaction
ML Value: Indicates transaction significance to buyer; differentiates strategic investments
```

**sale_price_per_employee** (Transaction Value Per Employee)
```
Formula: sale_price / number_of_employees  (or 0 if employees = 0)
Type: Numeric (continuous)
Meaning: Asset cost normalized by buyer business size
ML Value: Buyer capacity indicator; adjusts for business scale
```

##### D. Temporal Features

**contract_year** (Transaction Year)
```
Formula: EXTRACT(YEAR FROM contract_date)
Type: Numeric categorical ({2016, 2017, ..., 2024})
Meaning: Year of sale contract
ML Value: Captures temporal trends, market cycles, regulatory changes
```

**contract_month** (Transaction Month)
```
Formula: EXTRACT(MONTH FROM contract_date)
Type: Numeric categorical ({1, 2, ..., 12})
Meaning: Month of sale (seasonal patterns)
ML Value: Identifies seasonal buyer variation
Examples: Q4 year-end transactions vs. summer activity
```

**contract_quarter** (Transaction Quarter)
```
Formula: EXTRACT(QUARTER FROM contract_date)
Type: Numeric categorical ({1, 2, 3, 4})
Meaning: Business quarter of sale
ML Value: Aggregated seasonal pattern; links to fiscal/budget cycles
```

##### E. Business Lifecycle Features

**business_age_days_at_sale** (Buyer Business Age in Days)
```
Formula: contract_date - business_registration_date  (in days, clipped at 0)
Type: Numeric (continuous)
Meaning: Duration of buyer business existence at time of transaction
ML Value: Maturity indicator; established companies vs. startups may have different acquisition patterns
```

**business_age_years_at_sale** (Buyer Business Age in Years)
```
Formula: business_age_days_at_sale / 365.25
Type: Numeric (continuous)
Meaning: Business age in decimal years
ML Value: Easier interpretation than days; useful in regression or feature importance analysis
```

##### Summary Table: Feature Engineering Output

| Feature Name | Type | Source | Purpose |
|---|---|---|---|
| total_area_m2 | Numeric | Derived | Overall property scale |
| object_to_land_ratio | Numeric | Derived | Development intensity |
| price_per_total_m2 | Numeric | Derived | Price normalization |
| price_per_land_m2 | Numeric | Derived | Land value component |
| price_per_object_m2 | Numeric | Derived | Building value component |
| log_sale_price | Numeric | Derived | Log-normalized price |
| area_price_interaction | Numeric | Derived | Size-price interaction |
| capital_to_sale_price_ratio | Numeric | Derived | Buyer financial capacity |
| sale_price_per_employee | Numeric | Derived | Scale-adjusted buyer capacity |
| contract_year | Categorical | Temporal | Annual trends |
| contract_month | Categorical | Temporal | Seasonal patterns |
| contract_quarter | Categorical | Temporal | Quarterly cycles |
| business_age_days_at_sale | Numeric | Temporal | Business maturity (days) |
| business_age_years_at_sale | Numeric | Temporal | Business maturity (years) |
| is_land_only | Binary | Flag | Asset classification |
| has_object | Binary | Flag | Presence of structure |
| asset_structure_type | Categorical | Classification | Asset type summary |

**Outcome**: Dataset now contains 15+ interpretable, domain-meaningful features ready for ML

---

#### 2.8 Advanced Missing Value Handling

Rather than simple imputation, this section applies a **hierarchical, rule-based strategy** that respects data semantics and avoids introducing artificial bias.

##### Strategy Overview

```
Layer 1: Rule-Based Imputation
  ├─ Safe numeric columns → Fill with 0
  ├─ Activity fields → Fill with "Aktivitete tjera" (Other Activities)
  └─ Categorical fields → Fill with domain defaults
  
Layer 2: Group-Based Median Imputation
  ├─ Only for business fields
  ├─ Hierarchical grouping: Municipality × Sector × Business-Type
  ├─ Minimum group size: 5 observations
  └─ Conservative approach (prefer no fill over artificial fill)
  
Layer 3: Fallback Imputation
  ├─ Numeric → 0
  ├─ Date → Median or derived value
  └─ Text → Mode or safe default

Final: Verify zero missing values post-imputation
```

##### Layer 1: Rule-Based Imputation

**Numeric Fields – Zero Fill**
```
Columns: arbk_kapitali, arbk_numpunetoreve, cmimi_i_shitjes_se_asetit
Logic: Missing values assumed to represent zero quantity
Semantics: 0 capital = no registered capital, 0 employees = likely solo proprietor
Outcome: These fields never remain NULL
```

**Activity Fields – Default Category**
```
Columns: arbk_aktiviteti_1, arbk_aktiviteti_2, ... (any "aktiviteti" field)
Default: "Aktivitete tjera" (Other Activities)
Rationale: Missing activity likely means unclassified/other, not "no activity"
Outcome: Activity always defined, models can learn "other" pattern
```

**Categorical Defaults – Domain-Specific**
```
Business Name:           "Biznes i panjohur" (Unknown Business)
Business Type:           "Lloj biznesi i panjohur" (Unknown Business Type)  
Business Status:         "Status i panjohur" (Unknown Status)
Municipality:            "Komune e panjohur" (Unknown Municipality)
Category:                "Kategori e panjohur" (Unknown Category)
Sale Method:             "Menyre e panjohur" (Unknown Method)
Sale Type (New/Liquidation): "Te panjohura" (Unknown)

Rationale: Explicit "unknown" categories preserve information loss signal
Benefit: Model can learn patterns specifically around missing data
```

##### Layer 2: Group-Based Median Imputation (Business Fields Only)

Applied to columns with business significance:
- `arbk_kapitali` (business capital)
- `arbk_numpunetoreve` (employees)
- `arbk_pronari_1_kapitali` (owner 1 capital share)
- `arbk_pronari_1_kapitaliperqindje` (owner 1 capital percentage)

**Hierarchical Group Strategy** (in order of preference):

```
Level 1 Grouping: [Business Type, Municipality, Asset Category]
  Median calculation grouped by all three dimensions
  Minimum group size: 5 records
  If insufficient: Move to Level 2
  
Level 2 Grouping: [Business Type, Municipality]
  Median calculation grouped by these two
  Minimum group size: 5 records
  If insufficient: Move to Level 3
  
Level 3 Grouping: [Business Type]
  Median calculation grouped by business type only
  Minimum group size: 5 records
  If insufficient: Use fallback (zero)
```

**Conservative Approach**:
- Only fills groups with ≥5 observations (avoids tiny-sample bias)
- Tries multiple grouping levels (respects spatial/sectoral patterns)
- Falls back to zero rather than force-fill with unreliable estimates

**Example**:
```
Record: Business Type=Construction, Municipality=Tirana, Capital=NULL
Process:
  1. Look for median capital in [Construction, Tirana, Residential] groups
     → Found 8 records: median = 50,000 → USE THIS
  OR
  1. Look for median capital in [Construction, Tirana] groups
     → Found 4 records: TOO SMALL, skip
  2. Look for median capital in [Construction] groups
     → Found 27 records: median = 40,000 → USE THIS
  OR
  1. No reliable group → Fill with 0
```

##### Layer 3: Fallback Imputation

**For any remaining missing values after Layers 1-2:**

```
If numeric column:
  → Fill with 0
  → Record "fallback filled" in report
  
If date column:
  → Use dataset median date
  → If all nulls: Use system default (2000-01-01)
  
If text/categorical:
  → Use mode (most common value)
  → If no mode: Use safe default ("Unknown")
```

##### Verification

**Post-imputation validation**:
```python
remaining_nulls = df.isna().sum().sum()
assert remaining_nulls == 0, f"Found {remaining_nulls} remaining nulls"
```

**Outcome**: Final dataset contains ZERO missing values without introducing systematic bias

---

#### 2.9 Outlier Identification (Non-Modification)

In version v3.2, **outliers are identified but NOT modified**. This preserves data integrity and transparency.

**Method**: Interquartile Range (IQR)

```
For each numeric column:
  Q1 = 25th percentile
  Q3 = 75th percentile
  IQR = Q3 - Q1
  
  Lower Bound = Q1 - 1.5 × IQR
  Upper Bound = Q3 + 1.5 × IQR
  
  Outlier = (value < Lower Bound) OR (value > Upper Bound)
```

**Special Handling for Non-Negative Features**:
```
For inherently non-negative columns (price, area, capital, employees):
  Lower Bound = max(Q1 - 1.5 × IQR, 0)
  Rationale: Cannot have negative prices/areas; don't flag at 0
```

**Output**: For each flagged column, create binary outlier indicator
```
Example: price_per_total_m2_is_outlier_iqr
  Value: 1 if price_per_total_m2 is IQR outlier, else 0
  Type: Binary flag
  Purpose: Model explainability; can segment outlier transactions separately
```

**Benefits**:
- Preserves actual observed values (no artificial modification)
- Creates explicit outlier signals for model interpretation
- Enables downstream analysis: do outliers form distinct buyer segment?
- Maintains statistical validity for non-parametric models

---

#### 2.10 Feature Selection & Noise Reduction

Before training, low-information columns are removed to improve model efficiency and prevent overfitting.

##### Low-Information Column Removal

**Constant Columns** (nunique ≤ 1)
```
Definition: Columns where all values are identical or null
Action: REMOVE
Reason: Zero information content; cannot contribute to prediction
```

**Sparse Columns** (> 85% missing or same value)
```
Definition: More than 85% of rows contain same value or are null
Threshold: SPARSE_COL_THRESHOLD = 0.85
Action: REMOVE
Reason: Insufficient variance; overfitting risk; noise > signal
```

**Dominant-Value Columns** (≥ 85% one value, ≤ 3 unique values)
```
Definition: Single value represents ≥85% of rows, within ≤3 distinct values total
Action: REMOVE
Reason: Highly imbalanced; limited discriminative power
Example: "Status" field where 90% = "Active", 10% = "Inactive" → REMOVE
```

##### Data Leakage Column Removal

**Protected from Leakage**: Columns containing information not available at prediction time

```
Removed Columns:
- arbk_telefoni (business contact info)
- arbk_email (business contact info)
- arbk_webfaqja (business web presence)
- arbk_matchfield, arbk_matchscore (data enrichment metadata)
- arbk_datashuarjesbiznesit (business closure date – future information)
- arbk_emribiznesit_gjetur (enrichment artifact)

Reason: These fields are:
  1. Not available when predicting buyer for new asset
  2. Metadata about data quality/matching, not asset/business characteristics
  3. Indirect hints about business fate (closure date)
```

##### Identifier Column Removal

**Business identifiers** removed from model input:

```
- nr (row number)
- pak_id (asset ID)
- ndermarrja_shoqerore (company/business name)

Reason: 
- Contain no predictive signal
- Cause overfitting to specific IDs
- Necessitate separate storage of record keys in results
```

##### Date Column Archival

**Raw date columns** removed after temporal feature extraction:

```
Removed:
- data_e_kontrates (used to extract year/month/quarter/business_age_days)
- arbk_dataregjistrimit (used to compute business age)

Reason:
- Information preserved in engineered temporal features
- Continuous dates cause multicollinearity
- Easier for Random Forest to use discrete year/month than continuous date
```

##### Outcome

**Final feature set**:
- 40-60 clean, interpretable, ML-ready features (exact count varies by dataset)
- No sparse/constant/noisy columns
- No leakage
- No identifier columns
- No redundant date fields
- All features have clear business meaning

---

#### 2.11 Target Variable Preparation & Class Filtering

**Target**: `bleresi` (buyer type/category)

**Rare Class Removal**
```
Minimum Frequency Threshold: MIN_TARGET_FREQUENCY = 1
Logic: Remove rows where buyer type has < 1 occurrence in dataset
Practical Impact: Keeps all buyer types that appear at least once
Rationale: Ensures sufficient training examples per class
```

**Result**: Well-balanced buyer type distribution, sufficient examples per class

---

### 3: Final Cleaning & Quality Assurance

#### Missing Value Verification
```python
remaining_missing = df.isna().sum().sum()
assert remaining_missing == 0, f"Pipeline cannot proceed: {remaining_missing} missing values remain"
```

All records in final dataset have complete feature vectors.

#### Class Distribution Verification
```
Requirement: At least 2 distinct buyer types
Rationale: Binary or multi-class classification requires multiple classes
Action: Raise error if < 2 classes remain
```

---

### 4: Train-Test Split & Stratification

**Test Size**: 80% train, 20% test

**Stratification**: 
```
If minimum class frequency ≥ 2:
  Use stratified splitting to preserve class distribution
  Ensures train and test have similar buyer type proportions
  
Prevents:
  - Training on unrepresentative distribution
  - Test set containing rare classes not seen in training
```

**Randomization**: Fixed random seed (RANDOM_STATE=42) ensures reproducibility

---

## Final Output Artifacts

The complete pipeline generates 5 output files in the directories specified in `config.py`:

### 1. **cleaned_dataset_no_missing_vX.xlsx**
- **Location**: `data/processed/property_buyer/`
- **Content**: Complete clean dataset with all rows/features
- **Rows**: ~800-5000 (varies by source data)
- **Columns**: 40-60 clean, engineered features
- **Format**: Excel workbook (.xlsx), tab-separated
- **Usage**: Complete view of cleaned data; reference for analysis

### 2. **train_dataset_vX.xlsx**
- **Location**: `data/processed/property_buyer/`
- **Content**: Training set split with target variable
- **Rows**: ~80% of cleaned dataset
- **Format**: Excel workbook
- **Columns**: Feature columns + target (bleresi)
- **Usage**: Train Random Forest and other models

### 3. **test_dataset_vX.xlsx**
- **Location**: `data/processed/property_buyer/`
- **Content**: Test set split with target variable
- **Rows**: ~20% of cleaned dataset
- **Format**: Excel workbook
- **Columns**: Feature columns + target (bleresi)
- **Usage**: Evaluate model performance, estimate real-world accuracy


### 5. **preparation_report_vX.json**
- **Location**: `data/processed/property_buyer/`
- **Content**: Detailed statistics of all transformation stages
- **Format**: JSON
- **Key Metrics**:
  - Row counts: initial → after validation → after dedup → final
  - Columns: initial → sparse removed → constant removed → final
  - Missing values: initial → strategy → final (should be 0)
  - Outliers: count and distribution by feature
  - Features: count before/after selection, selected feature names
  - Target: class distribution, buyers kept/removed
  - Model: feature importance, selected features
- **Usage**: Audit trail; reproducibility documentation; performance baseline

---


---

### Field Documentation: Derived Area & Price Features

| Field Name | Type | Description | Formula | Business Meaning | ML Relevance |
|---|---|---|---|---|---|
| `total_area_m2` | Numeric (float) | Combined property area | object_area + land_area | Total usable/owned footprint | Comprehensive size indicator; primary valuation driver |
| `object_to_land_ratio` | Numeric (float) | Building intensity ratio | object_area / land_area (or 0) | Proportion of property that is developed | Distinguishes land speculation from improvement-focused investment |
| `price_per_total_m2` | Numeric (float) | Price per unit total area | sale_price / total_area_m2 | Unit price intensity | Normalized pricing comparison; market heat indicator |
| `price_per_land_m2` | Numeric (float) | Price per unit land area | sale_price / land_area_m2 | Land unit pricing | Land-focused valuation; pure land investment metric |
| `price_per_object_m2` | Numeric (float) | Price per unit building area | sale_price / object_area_m2 | Building unit pricing | Construction cost proxy; quality/finishability indicator |
| `log_sale_price` | Numeric (float) | Log-transformed price | log(1 + sale_price) | Normalized price distribution | Reduces right-skew in price data; stabilizes model variance |
| `is_high_value_asset` | Binary (0/1) | Price above population median | 1 if sale_price > median else 0 | Premium vs. standard market segment | Buyer sophistication/capital segmentation |
| `area_price_interaction` | Numeric (float) | Size-price compound metric | total_area_m2 × price_per_total_m2 | Captures size-intensity combinations | Non-linear relationship; large premium properties vs. others |

---

### Field Documentation: Business Dynamics Features

| Field Name | Type | Description | Formula | Business Meaning | ML Relevance |
|---|---|---|---|---|---|
| `capital_to_sale_price_ratio` | Numeric (float) | Buyer capital to transaction value | buyer_capital / sale_price (or 0) | Transaction significance to buyer | Indicates whether purchase is major or minor decision for buyer |
| `sale_price_per_employee` | Numeric (float) | Transaction cost per employee | sale_price / num_employees (or 0) | Per-employee investment | Employee capacity normalized; scalability assumption |
| `business_age_days_at_sale` | Numeric (integer) | Buyer business age in days | contract_date - registration_date | Business maturity in days | Startup vs. established enterprise distinction |
| `business_age_years_at_sale` | Numeric (float) | Buyer business age in years | business_age_days_at_sale / 365.25 | Business maturity in years | Experience factor; established companies vs. new ventures |

---

### Complete Field Count Summary

| Category | Field Count | Type Distribution |
|---|---|---|
| Original Asset Fields | 10 | 3 numeric, 7 categorical |
| ARBK Business Fields | 13 | 5 numeric, 8 categorical |
| Derived Area/Price Features | 8 | 8 numeric |
| Business Dynamics Features | 4 | 4 numeric |
| Temporal Features | 3 | 3 numeric |
| Asset Classification Features | 5 | 2 binary, 1 categorical, 2 numeric/flag |
| Outlier Flags | Variable | Binary (1 per outlier-prone numeric column) |
| Target Variable | 1 | 1 categorical (multiclass) |
| **TOTAL (Approximate)** | **~45-60** | ~30 numeric, ~15 categorical, binary flags |

---

## Visualization
All the visualization below are taken from the output of the code residing from `notebooks/01_data_exploration.ipynb`

### 1. **Data Types**
**Purpose**: Visualize the distribution of data types across the dataset

**Visualization Output**:
- Bar chart: Number of columns by data type (Numeric, Text/Object, Date/Time)

**Output File**: `visualizations/Data_Types_Distribution.png`
![Data_Types_Distribution.png](visualizations/Data_Types_Distribution.png)

---

### 2. **Missing Values**
**Purpose**: Visualize the percentage of missing values in each column

**Visualization Output**:
- Horizontal bar chart: Percentage of null values per column with color-coded severity (red >50%, orange 20-50%, blue <20%)

**Output File**: `visualizations/Missing_Values_Percentage.png`
![Missing_Values_Percentage.png](visualizations/Missing_Values_Percentage.png)

---

### 3. **ARBK Match Rate**
**Purpose**: Visualize the match rate between buyer names and ARBK business registry

**Visualization Output**:
- Pie chart: Percentage of records successfully matched vs. unmatched to ARBK database

**Output File**: `visualizations/ARBK_Match_Rate.png`
![ARBK_Match_Rate.png](visualizations/ARBK_Match_Rate.png)

---


### 4. **Price plot**
**Purpose**: Visualize the plot of Price

**Visualization Output**:
- Bar chart: the Price of sold properties

**Output File**: `visualizations/Price_histogram_plot.png`
![Price_histogram_plot.png](visualizations/Price_histogram_plot.png)
---
### 5. **Price plot logarithmic**
**Purpose**: Visualize the plot of Price

**Visualization Output**:
- Bar chart: the Price of sold properties logarithm applied

**Output File**: `visualizations/Price_histogram_plot_log.png`
![Price_histogram_plot_log.png](visualizations/Price_histogram_plot_log.png)
---
### 6. **land + built object in square meters histogram plot**
**Purpose**: Visualize the plot of land + built object in square meters

**Visualization Output**:
- Bar chart: the land square meters plus  the object built on top of the land square meters

**Output File**: `visualizations/land_and_object_histogram_plot.png`
![land_and_object_histogram_plot.png](visualizations/land_and_object_histogram_plot.png)
---
### 7. **land + built object in square meters histogram plot logarithmic**
**Purpose**: Visualize the plot of land + built object in square meters logarithm applied

**Visualization Output**:
- Bar chart: the land square meters plus  the object built on top of the land square meters

**Output File**: `visualizations/land_and_object_histogram_plot_log.png`
![land_and_object_histogram_plot_log.png](visualizations/land_and_object_histogram_plot_log.png)
---


### Visualization Scripts Summary Table

| Script Name | Input | Output | Key Metric | Business Value |
|---|---|---|---|---|
| plot_missing_values.py | raw + cleaned data | PNG (1200×800) | Missing % per column | Data quality audit |
| plot_target_distribution.py | cleaned_dataset + train/test | PNG (1400×600) | Class balance | Model fairness check |
| plot_asset_structure_distribution.py | cleaned_dataset | PNG (1400×700) | Asset type % | Market composition |
| plot_price_distribution.py | cleaned_dataset | PNG (1600×900) | Price distribution shape | Normalization validation |
| plot_area_distribution.py | cleaned_dataset | PNG (1600×900) | Area statistics | Data quality check |
| plot_price_per_m2_distribution.py | cleaned_dataset | PNG (1600×900) | Unit pricing patterns | Location premium analysis |
| plot_correlation_heatmap.py | cleaned_dataset (numeric only) | PNG (1400×1400) | Correlation matrix | Multicollinearity check |
| plot_outlier_summary.py | cleaned_dataset + outlier flags | PNG (1600×900) | Outlier counts & rates | Data extremity analysis |
| plot_top_buyers.py | cleaned_dataset | PNG (1400×700) | Top buyer types | Market concentration |
| plot_feature_importance.py | trained model + dataset | PNG (1200×700) | Feature importance scores | Model interpretability |

---

## How to Run the Pipeline

### Prerequisites
- Python 3.8+
- Dependencies: pandas, numpy, scikit-learn, joblib, openpyxl
- Raw dataset: `sales_with_converted_rents_enriched_with_arbk.xlsx`

### Installation
```bash
# Install dependencies
pip install pandas numpy scikit-learn joblib openpyxl

# Or use requirements.txt (if provided)
pip install -r requirements.txt
```

### Running the Pipeline

#### Command Line
```bash
# From repository root
cd ../machineLearning-2026-Gr12

# Run the ETL pipeline
python -m src.property_buyer_pipeline.pipeline
```

### Configuration

Edit `src/property_buyer_pipeline/config.py` to customize:

```python
# Data paths
INPUT_FILE = "sales_with_converted_rents_enriched_with_arbk.xlsx"
OUTPUT_DIR = Path("data/processed/property_buyer")
MODELS_DIR = Path("data/raw/models/property_buyer")

# Quality thresholds
SPARSE_COL_THRESHOLD = 0.85  # Remove columns > 85% missing
DOMINANT_VALUE_THRESHOLD = 0.85  # Remove columns where 1 value > 85%


```

## Citation & Attribution

If you use this dataset or pipeline in research or publication, please reference:

```
Property Buyer Prediction Dataset & ETL Pipeline. 
Machine Learning Project (2026).
Repository: https://github.com/yourusername/machineLearning-2026-Gr12
```