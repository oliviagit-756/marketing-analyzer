import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def generate_campaign(row):
    segment = row["Segment"]
    product = row["Top_Product"]

    if pd.isna(product):
        product = "your favorite products"

    if segment == "Champions / VIP Customers":
        campaign_goal = "Reward and retain high-value customers"
        message = f"Thank you for being one of our top customers! Enjoy an exclusive VIP reward on {product}."

    elif segment == "Loyal Customers":
        campaign_goal = "Increase repeat purchases"
        message = f"We appreciate your loyalty! Get a special offer on {product}."

    elif segment == "Occasional Buyers / Need Attention":
        campaign_goal = "Encourage more frequent purchases"
        message = f"Still interested in {product}? Here is a limited-time personalized offer just for you."

    elif segment == "Inactive / Lost Customers":
        campaign_goal = "Win back inactive customers"
        message = f"We miss you! Come back and enjoy a special discount on {product}."

    else:
        campaign_goal = "General engagement"
        message = "Explore our latest offers selected specially for you."

    return pd.Series([campaign_goal, message])

def run_customer_segmentation(df):
    df = df.copy()

    # Cleaning
    df = df.dropna(subset=["CustomerID"])
    df = df.dropna(subset=["Description"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # Type conversion
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

    # Total amount
    df["Total"] = df["Quantity"] * df["UnitPrice"]

    # Reference date
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # RFM
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "InvoiceNo": "nunique",
        "Total": "sum"
    })

    rfm.rename(columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "Total": "Monetary"
    }, inplace=True)

    rfm = rfm.reset_index()

    # Top product
    top_product = df.groupby(["CustomerID", "Description"])["Quantity"].sum().reset_index()

    top_product = top_product.sort_values(
        ["CustomerID", "Quantity"],
        ascending=[True, False]
    )

    top_product = top_product.drop_duplicates("CustomerID")

    top_product = top_product[["CustomerID", "Description"]]
    top_product.rename(columns={"Description": "Top_Product"}, inplace=True)

    # Merge RFM + Top Product
    customer_features = rfm.merge(top_product, on="CustomerID", how="left")

    # K-Means
    X = customer_features[["Recency", "Frequency", "Monetary"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    customer_features["Cluster"] = kmeans.fit_predict(X_scaled)

    # Segment mapping
    segment_map = {
        2: "Champions / VIP Customers",
        3: "Loyal Customers",
        0: "Occasional Buyers / Need Attention",
        1: "Inactive / Lost Customers"
    }

    customer_features["Segment"] = customer_features["Cluster"].map(segment_map)

    # Campaign messages
    customer_features[["Campaign_Goal", "Campaign_Message"]] = customer_features.apply(
        generate_campaign,
        axis=1
    )

    return customer_features