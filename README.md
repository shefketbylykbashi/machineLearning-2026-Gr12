# Machine Learning - Project I: Potential Buyer Recommendation System

## Project Overview

This project aims to build a machine learning system that recommends potential buyers for properties when they are put up for sale. The system combines real estate sales data from Kosovo's Agency for Privatization with business information from ARBK.

## Objective

When a property is listed for sale, suggest potential buyers based on:
- Sales data from the Agency of Privatization of Kosovo
- Business profiles and information from ARBK
- Data enrichment by linking business names with ARBK records

## Data Sources

1. **Agency of Privatization of Kosovo (PAK)**
   - Real estate sales data: prices, locations, property types, buyers
   - Website: https://www.pak-ks.org/page.aspx?id=1,37

2. **Agjencia e Regjistrimit te Bizneseve (ARBK)**
   - Business information linked to buyer names
   - Enriches the dataset with additional business context
   - Website: https://arbk.rks-gov.net/

## Project Structure

```
├── README.md                                    # Project documentation
├── requirements.txt                             # Python dependencies
├── data/
│   └── raw/
│       └── sales_with_converted_rents_enriched_with_arbk.xlsx
├── notebooks/
│   └── 01_data_exploration.ipynb               # Data exploration and quality assessment
└── scripts/
    └── data_preparation/
        ├── convert_rent_to_sales.py            # Data transformation script
        └── enrich_with_arbk.py                 # ARBK enrichment script
```

## Current Status

**Phase**: Data Planning & Exploration
- Dataset collection
- Initial data enrichment with ARBK
- Data exploration and quality analysis


### Installation

1. **Clone or download the project**
   ```bash
   cd machineLearning-2026-Gr12
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Using the Project

1. **Open Jupyter Notebooks**
   - Open `notebooks/01_data_exploration.ipynb`
   - Select the `.venv` as the Jupyter kernel

3. **Explore the Data**
   - Run the data exploration notebook to understand dataset characteristics:
     - Data types and distributions
     - Missing values (nulls)
     - Data quality metrics (duplicates, ARBK match rate)
     - Categorical cardinality analysis


## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **openpyxl**: Excel file handling

See `requirements.txt` for complete list and versions.

## Key Findings (so far)

### Dataset Overview
- **Total Records**: 3,483 property sales
- **Total Features**: 47 columns (numeric, categorical, date/time)
- **Data Types**: Mix of integers, floats, objects, and datetime columns

### Data Quality
**No Duplicate Rows**: 100% unique records (0 duplicates)

### ARBK Matching Success
The system successfully linked business names to ARBK records:
- **Matched**: 2,687 records (77.1%)
- **Not Matched**: 796 records (22.9%)

**Matching Methods Used**:
- Exact owner name match: 1,856 records
- Fuzzy owner name match: 476 records
- Exact business name match: 232 records
- Fuzzy business name match: 123 records

### Data Characteristics
- **Missing Values**: Present in several columns (see notebook for detailed analysis)
- **Categorical Cardinality**: Analyzed to identify low-cardinality columns (<= 20 unique values) suitable for feature engineering
- **Visualizations**: Data types distribution, null values heatmap, ARBK match rate pie chart, cardinality bar charts

*See `notebooks/01_data_exploration.ipynb` for complete tables, visualizations, and detailed analysis.*

