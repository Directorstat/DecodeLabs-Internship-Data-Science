# Customer Segmentation Using PCA and K-Means Clustering

A machine learning project that applies unsupervised learning techniques to discover
hidden customer groups in unlabeled retail transaction data. Using Principal Component
Analysis (PCA) for dimensionality reduction and K-Means clustering for segmentation,
this project translates raw retail transactions into three actionable business personas
that can drive targeted marketing, retention strategies, and revenue growth decisions.

---

## Problem Statement

Retail businesses collect thousands of transaction records daily but rarely know who
their customers truly are beyond basic demographics. Without understanding behavioural
patterns, marketing budgets are wasted treating all customers the same way. A Champion
customer spending 8,000 per year deserves different treatment than a lapsed customer
who bought once five months ago and never returned. This project uses pure statistical
logic to discover these hidden groups automatically, without any predefined labels.

---

## Objectives

- Engineer meaningful customer behavioural features from raw transaction data
- Apply PCA to reduce 19 features into a lower-dimensional space while retaining
  maximum information
- Use the Elbow Method and Silhouette Score to mathematically prove the optimal
  number of clusters
- Train a K-Means model to segment customers into distinct groups
- Translate mathematical clusters into actionable business personas with clear
  strategic recommendations

---

## Dataset

- Source: Kaggle
- Name: Online Retail Dataset
- Link: https://www.kaggle.com/datasets/vijayuv/onlineretail/data
- Size: 541,909 transaction rows, 8 columns
- Coverage: December 2010 to December 2011, UK-based online retailer
- Customers after cleaning: 4,338 unique customers

### Raw Columns

| Column | Description |
|---|---|
| InvoiceNo | Transaction identifier. Starts with C for cancellations |
| StockCode | Product identifier |
| Description | Product name |
| Quantity | Units purchased per line item |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per unit in GBP |
| CustomerID | Unique customer identifier |
| Country | Customer country |

---

## Methodology

### 1. Data Cleaning
- Removed 135,080 rows with missing CustomerID (25% of raw data)
- Removed cancelled invoices where InvoiceNo started with letter C
- Removed rows with negative or zero Quantity values
- Removed rows with zero or negative UnitPrice
- Converted InvoiceDate from text to datetime format
- Created TotalPrice column by multiplying Quantity by UnitPrice
- Final clean dataset: 397,884 transaction rows across 4,338 customers

### 2. Feature Engineering
Built 19 customer-level behavioural features from transaction rows:

| Category | Features |
|---|---|
| Recency and Timing | Recency, FirstPurchaseDays, PurchaseSpan, UniqueInvoiceDays, AvgDaysBetweenPurchases |
| Transaction Volume | Frequency, TotalTransactionLines, TotalQuantity, AvgQuantityPerOrder |
| Spending Patterns | Monetary, AvgOrderValue, MaxSingleTransaction, MinSingleTransaction, StdOrderValue, OrderValueGrowth |
| Product Diversity | UniqueProducts, RevenuePerProduct |
| Pricing Behaviour | AvgUnitPrice, PurchaseConsistency |

### 3. Preprocessing
- Applied log1p transformation to 17 skewed features to compress extreme values
- Applied StandardScaler to normalise all features to mean=0 and std=1
- This prevents high-magnitude features from dominating PCA directions

### 4. Dimensionality Reduction (PCA)
- Fitted PCA on all 19 scaled features
- PC1 alone captures 39.93% of total variance
- PC2 adds 25.57%, bringing the two-component total to 65.50%
- 7 components retain 93.66% of all customer behavioural information
- 2 components retain 65.50% and are used for 2D cluster visualisation
- Final clustering performed in 7-component PCA space for accuracy

### 5. Optimal K Selection
Used two complementary methods to mathematically prove K=3:

Elbow Method: Inertia drops steeply from K=2 to K=4, then begins to flatten,
indicating diminishing returns beyond K=3.

Silhouette Score: K=2 scores 0.3007 but produces only two broad groups with
limited business utility. K=3 scores 0.2779, the second highest score, while
producing three meaningfully distinct and actionable customer segments.

### 6. K-Means Clustering
- Algorithm: K-Means with K=3, n_init=10, random_state=42
- Input: 7-component PCA matrix (4338 rows x 7 columns)
- Final Silhouette Score: 0.2779

---

## Results

### Cluster Distribution

| Persona | Customers | Percentage |
|---|---|---|
| Loyal Regular Customers | 1,973 | 45.5% |
| At Risk / Lapsed Customers | 1,716 | 39.6% |
| Champion Customers | 649 | 15.0% |

### Cluster Profiles

| Persona | Recency (days) | Frequency | Monetary (GBP) | Avg Order Value | Unique Products |
|---|---|---|---|---|---|
| Champion Customers | 54.83 | 10.00 | 7,987.56 | 335.70 | 75.33 |
| Loyal Regular Customers | 53.99 | 5.09 | 1,609.32 | 15.22 | 91.83 |
| At Risk / Lapsed Customers | 151.12 | 1.17 | 321.85 | 28.33 | 21.40 |

---

## Business Personas and Recommendations

### Champion Customers (649 customers, 15%)
These are your highest-value customers. They spend nearly 8,000 per year, order 10
times, and engage across 75 unique products. Despite representing only 15% of the
customer base, they generate disproportionate revenue.

Recommended Actions:
- Enrol in a premium VIP loyalty programme with exclusive rewards
- Provide early access to new product launches
- Assign dedicated account managers for top spenders
- Send personalised thank-you communications and anniversary offers

### Loyal Regular Customers (1,973 customers, 45.5%)
The largest and most stable segment. They order 5 times per year and browse the
widest product range of any group (91 unique products) but spend modestly per order
at just 15.22 average order value.

Recommended Actions:
- Target with bundle offers and minimum spend thresholds to increase order value
- Use cross-sell recommendations based on their broad browsing patterns
- Introduce tiered loyalty rewards that incentivise higher spend per transaction
- A/B test promotional campaigns aimed at moving 10% of this group into Champion tier

### At Risk / Lapsed Customers (1,716 customers, 39.6%)
The most urgent retention challenge. Last purchase was 151 days ago on average.
Most made only one purchase and disappeared within 7.61 days of their first order.

Recommended Actions:
- Launch automated win-back email campaigns with time-limited discount codes
- Reference their previous purchases to create personalised reactivation messages
- Offer a low-barrier incentive such as free shipping on the next order
- Flag customers exceeding 180 days recency for final win-back attempt before
  removing from active marketing lists

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Preprocessing | Scikit-learn StandardScaler |
| Dimensionality Reduction | Scikit-learn PCA |
| Clustering | Scikit-learn KMeans |
| Evaluation | Scikit-learn Silhouette Score |
| Visualisation | Matplotlib, Seaborn |
| Model Saving | Joblib |
| Development | Google Colab |

---

## Project Structure
customer-segmentation/

├── customer_segmentation.ipynb       # Full project notebook

├── kmeans_customer_segmentation.pkl  # Trained KMeans model

├── pca_7_components.pkl              # 7-component PCA model

├── pca_2_components.pkl              # 2-component PCA for visualisation

├── customer_scaler.pkl               # Fitted StandardScaler

├── persona_map.json                  # Cluster to persona name mapping

├── customer_segments.csv             # Final segmented customer table

└── README.md                         # Project documentation
---

## How to Run Locally

1. Clone the repository

```bash
git clone https://github.com/Directorstat/Task3.git
cd DecodeLabs-Internship
```

2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

3. Download the dataset from Kaggle
4. Open and run the notebook

```bash
jupyter notebook customer_segmentation.ipynb
```

---

## Key Findings

- PC1 alone captures 39.93% of all customer behavioural variance, confirming a
  dominant axis of difference exists between high-value and low-value customers
- 7 PCA components retain 93.66% of all information, reducing dimensionality from
  19 to 7 while losing only 6.34% of customer signal
- K=3 was selected using both the Elbow Method and Silhouette Score analysis,
  producing a final silhouette score of 0.2779
- Champion Customers (15% of base) generate approximately 60% of total revenue
  based on their average monetary values relative to other segments
- 39.6% of customers are At Risk or Lapsed, representing a significant retention
  and re-engagement opportunity for the business
- Loyal Regular Customers browse the widest product range (91 unique products)
  despite having the lowest average order value, making them ideal targets for
  upsell and bundle campaigns

---

## Author

Waliyullahi Abdulrouf
Data Scientist | 3MTT Nigeria Fellow | NYSC Corps Member
GitHub: github.com/Directorstat
LinkedIn: linkedin.com/in/waliyullahi-abdulrouf-863145231

---

## Acknowledgement

This project was developed as part of the Decode Labs Data Science Internship program
under the Technical Mastery Phase: Unsupervised Learning track. Dataset sourced from
the UCI Machine Learning Repository.

---

## Disclaimer

All customer data used in this project is publicly available and anonymised. No
personally identifiable information was used or exposed at any stage of this analysis.