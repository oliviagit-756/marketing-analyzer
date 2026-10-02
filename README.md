# ML Marketing tool

An AI-powered customer segmentation and personalized campaign generation tool built using **Python**, **K-Means Clustering**, and **Streamlit**.

This project analyzes customer transaction data, identifies customer behavior patterns using RFM analysis, segments customers into meaningful groups, and generates personalized marketing campaign messages for each segment.

---

## Project Overview

Businesses often collect customer purchase data but struggle to convert that data into useful marketing actions. Many companies send the same campaign to all customers, even though customer behavior differs from person to person.

This project solves that problem by analyzing customer purchase behavior and generating personalized campaign suggestions based on customer segments.

The system follows this workflow:

```text
Customer Transaction Data
        ↓
Data Cleaning
        ↓
RFM Analysis
        ↓
K-Means Clustering
        ↓
Customer Segmentation
        ↓
Personalized Campaign Generation
        ↓
Streamlit Dashboard
```

---

## Features

* Upload customer transaction CSV file
* Clean missing and invalid data
* Calculate RFM values:

  * Recency
  * Frequency
  * Monetary
* Segment customers using K-Means Clustering
* Assign business-friendly segment names
* Generate personalized campaign messages
* Display interactive dashboard using Streamlit
* Visualize customer segments using Plotly charts
* Download final campaign output as CSV

---

## Customer Segments

The project groups customers into the following segments:

| Segment                            | Meaning                                             |
| ---------------------------------- | --------------------------------------------------- |
| Champions                          | Recent buyers with high frequency and high spending |
| Loyal Customers                    | Regular customers with good purchase behavior       |
| Occasional Buyers                  | Customers who buy sometimes but need engagement     |
| Inactive Customers                 | Customers who have not purchased for a long time    |

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* K-Means Clustering
* StandardScaler
* Plotly
* Streamlit

---

## Dataset Requirements

The input CSV file should contain the following columns:

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

The application calculates the `Total` column using:

```python
Total = Quantity * UnitPrice
```

---

## How the Project Works

### 1. Data Cleaning

The dataset is cleaned by:

* Removing rows with missing `CustomerID`
* Removing rows with missing `Description`
* Removing invalid transactions where `Quantity <= 0`
* Removing invalid rows where `UnitPrice <= 0`
* Converting `InvoiceDate` into proper datetime format

### 2. RFM Analysis

RFM analysis is used to understand customer behavior.

| Feature   | Meaning                             |
| --------- | ----------------------------------- |
| Recency   | How recently the customer purchased |
| Frequency | How often the customer purchased    |
| Monetary  | How much money the customer spent   |

### 3. Customer Segmentation

K-Means Clustering is applied on the RFM values to group customers with similar behavior.

Before applying K-Means, the RFM features are scaled using `StandardScaler` so that one feature does not dominate the clustering result.

### 4. Campaign Generation

After segmentation, the system generates personalized campaign messages based on:

* Customer segment
* Top purchased product
* Campaign goal

Example:

```text
Segment: Inactive / Lost Customers
Campaign Goal: Win back inactive customers
Message: We miss you! Come back and enjoy a special discount on your favorite product.
```

---

## Dashboard

The Streamlit dashboard includes:

* Dataset preview
* Key metrics
* Customer segment distribution
* RFM segment summary
* Customer behavior scatter plot
* Personalized campaign table
* Downloadable campaign CSV

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ai-personalized-marketing-tool.git
```

Move into the project folder:

```bash
cd ai-personalized-marketing-tool
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```
---

## Output

The final output contains:

* Customer ID
* Customer Segment
* Top Product Purchased
* Campaign Goal
* Personalized Campaign Message

The output can be downloaded as:

```text
personalized_marketing_campaigns.csv
```

---

## Interview Explanation

This project uses customer transaction data to perform customer segmentation and generate personalized marketing campaigns. I first cleaned the dataset, then created RFM features to understand customer behavior. Since the data did not contain predefined customer labels, I used K-Means Clustering as an unsupervised machine learning technique. After clustering, I interpreted each cluster based on average Recency, Frequency, and Monetary values and generated personalized campaign messages for each customer segment.

---

## Future Improvements

* Add email campaign integration
* Use a Generative AI model for more advanced campaign text generation
* Add customer lifetime value prediction
* Add product recommendation system
* Deploy the app on Streamlit Cloud
* Add database support for real-time customer data

---

## Author

Developed as an AI and Data Science project to demonstrate customer segmentation, machine learning, and business-focused dashboard development.

