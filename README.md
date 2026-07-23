# SmartCart – Customer Segmentation System

SmartCart is an unsupervised machine learning project that segments e-commerce customers into meaningful groups based on their purchasing behaviour, engagement, and demographics. The goal is to replace generic, one-size-fits-all marketing with **data-driven, cluster-specific strategies** for personalised marketing and customer retention.

## Problem Statement

SmartCart is a growing e-commerce platform serving customers across multiple countries. It currently applies the same marketing and engagement strategy to every customer, which leads to inefficient spend, missed opportunities to retain high-value customers, and delayed identification of churn-prone users.

This project builds an intelligent customer segmentation system using **clustering algorithms** to discover hidden patterns in customer behaviour from historical transaction data, grouping customers by purchasing behaviour, engagement level, and loyalty indicators.

## Project Structure

```
smartcart-clustering-system/
├── smartcart.ipynb                     # Main notebook: preprocessing, EDA, clustering
├── smartcart_customers.csv             # Raw dataset (2240 customers, 22 attributes)
├── SmartCart_Clustering_System.pdf     # Problem statement & dataset spec
└── README.md
```

## Dataset

`smartcart_customers.csv` contains **2,240 customer records** with **22 attributes**, grouped into four categories:

**Demographics**
| Feature | Description |
|---|---|
| `ID` | Unique customer identifier |
| `Year_Birth` | Year of birth |
| `Education` | Highest education level achieved |
| `Marital_Status` | Marital status |
| `Income` | Yearly household income |
| `Kidhome` | Number of small children in household |
| `Teenhome` | Number of teenagers in household |
| `Dt_Customer` | Date customer enrolled with SmartCart |

**Purchase Behaviour – Amount Spent**
| Feature | Description |
|---|---|
| `MntWines` | Amount spent on wine |
| `MntFruits` | Amount spent on fruits |
| `MntMeatProducts` | Amount spent on meat |
| `MntFishProducts` | Amount spent on fish |
| `MntSweetProducts` | Amount spent on sweets |
| `MntGoldProds` | Amount spent on gold products |

**Purchase Behaviour – Frequency**
| Feature | Description |
|---|---|
| `NumDealsPurchases` | Purchases made using discounts |
| `NumWebPurchases` | Purchases made through the website |
| `NumCatalogPurchases` | Purchases made through catalog |
| `NumStorePurchases` | Purchases made in physical stores |
| `NumWebVisitsMonth` | Website visits per month |

**Feedback & Other**
| Feature | Description |
|---|---|
| `Recency` | Days since last purchase |
| `Complain` | Complained in last 2 years (1 = Yes, 0 = No) |
| `Response` | Responded to last campaign (1 = Yes, 0 = No) |

## Workflow

The notebook (`smartcart.ipynb`) follows this pipeline:

1. **Load Data** – reads `smartcart_customers.csv` (2240 rows × 22 columns).
2. **Missing Values** – `Income` has 24 missing values, imputed with the **median**.
3. **Feature Engineering**
   - `Age` = 2026 − `Year_Birth`
   - `Customer_Tenure_Days` = days between `Dt_Customer` and the most recent enrollment date in the dataset
   - `Total_Spending` = sum of all `Mnt*` product spend columns
   - `Total_Children` = `Kidhome` + `Teenhome`
   - `Education` simplified into 3 tiers: **Undergraduate** (Basic, 2n Cycle), **Graduate** (Graduation), **Postgraduate** (Master, PhD)
   - `Living_With` derived from `Marital_Status`: **Partner** (Married, Together) vs. **Alone** (Single, Divorced, Widow, Absurd, YOLO)
4. **Drop Columns** – removes `ID`, `Year_Birth`, `Marital_Status`, `Kidhome`, `Teenhome`, `Dt_Customer`, and the raw `Mnt*` spend columns (now captured by `Total_Spending`) → 2240 rows × 15 columns.
5. **Outlier Removal** – pairplot-based inspection, then filters `Age < 90` and `Income < 600,000` → 2236 rows retained (4 removed).
6. **Correlation Heatmap** – inspects relationships among numeric features.
7. **Encoding** – `OneHotEncoder` applied to `Education` and `Living_With` → 2236 rows × 18 columns.
8. **Scaling** – `StandardScaler` applied to all features before clustering.
9. **Dimensionality Reduction** – `PCA` reduces the feature space to **3 components** for clustering and visualization (explained variance ≈ 23.2%, 11.4%, 10.4% for PC1–PC3), visualized in a 3D scatter plot.
10. **Choosing K**
    - **Elbow Method** (WCSS) via `KneeLocator` → optimal K = **4**
    - **Silhouette Score** computed for K = 2–10 as a cross-check
11. **Clustering** – two algorithms compared on the PCA-reduced data:
    - `KMeans` (K = 4)
    - `AgglomerativeClustering` (K = 4, Ward linkage) — used for final cluster characterization
12. **Cluster Characterization**
    - Cluster size distribution (count plot)
    - Income vs. Total Spending scatter plot, colored by cluster
    - Per-cluster feature averages (`groupby("cluster").mean()`)

## Cluster Summary (Agglomerative Clustering, K = 4)

| Cluster | Avg. Income | Avg. Recency | Web Purchases | Catalog Purchases | Store Purchases | Web Visits/Month | Complaint Rate |
|---|---|---|---|---|---|---|---|
| 0 | ~$39,681 | ~48.9 days | 3.15 | 0.97 | 4.14 | 6.31 | 1.1% |
| 1 | ~$72,808 | ~49.2 days | 5.69 | 5.50 | 8.66 | 3.58 | 0.6% |
| 2 | ~$36,960 | ~48.3 days | 2.71 | 0.84 | 3.62 | — | — |
| 3 | ~$70,723 | ~50.5 days | 5.79 | — | — | — | — |

**Pattern:** two clusters (1 & 3) represent **higher-income, high-engagement customers** who buy heavily through catalog and store channels and visit the website less often — likely SmartCart's premium/loyal segment. The other two (0 & 2) represent **lower-income, more web-browsing-oriented customers** with fewer catalog/store purchases — a segment more responsive to online deals and offers.

*(Full per-cluster averages for all features are available in the notebook's final `cluster_summary` output.)*

## Tech Stack

- **Python 3**
- **pandas** – data handling
- **seaborn**, **matplotlib** – visualization (including 3D plots)
- **scikit-learn** – preprocessing (`OneHotEncoder`, `StandardScaler`), `PCA`, `KMeans`, `AgglomerativeClustering`, `silhouette_score`
- **kneed** – automatic elbow-point detection for the Elbow Method

## Getting Started

### Prerequisites
```bash
pip install pandas matplotlib seaborn scikit-learn kneed jupyter
```

### Run the Project
```bash
git clone <your-repo-url>
cd smartcart-clustering-system
jupyter notebook smartcart.ipynb
```
Run all cells in order from top to bottom.

## Known Issues / Next Steps

- Both `KMeans` and `AgglomerativeClustering` are fit, but only the Agglomerative labels are carried into the final characterization step — consider comparing both with silhouette scores before picking a final model.
- `Response` (campaign response) and `Complain` are available but not deeply explored in cluster profiling — worth adding to understand churn-risk vs. engaged segments per cluster.
- Cluster interpretation/naming (e.g. "high-value loyal", "price-sensitive online") is not yet finalized in the notebook — add descriptive business-facing labels to each cluster for stakeholders.
- No model persistence step yet (e.g. saving the fitted `scaler`, `pca`, and cluster model via `joblib`) for scoring new customers.
- PCA explained variance for 3 components is only ~45% combined — consider evaluating clustering directly on scaled features (or more PCA components) to check if segmentation quality improves.
