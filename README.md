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