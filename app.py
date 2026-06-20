import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Personalized Marketing Tool",
    page_icon="📊",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        border: 1px solid #eef2f7;
        text-align: center;
    }

    .metric-label {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #111827;
    }

    .info-box {
        background-color: #f3f4f6;
        padding: 18px;
        border-radius: 15px;
        border-left: 5px solid #2563eb;
        margin-top: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- HELPER FUNCTIONS ----------------
def clean_product_name(product):
    if pd.isna(product):
        return "your favorite products"

    product = str(product).strip().title()

    if len(product) > 45:
        product = product[:45] + "..."

    return product


def generate_campaign(row):
    segment = row["Segment"]
    product = clean_product_name(row["Top_Product"])

    if segment == "Champions ":
        campaign_goal = "Reward and retain high-value customers"
        message = f"Thank you for being one of our top customers! Enjoy an exclusive VIP reward on {product}."

    elif segment == "Loyal Customers":
        campaign_goal = "Increase repeat purchases"
        message = f"We appreciate your loyalty! Get a special offer on {product} and continue enjoying your favorite picks."

    elif segment == "Occasional Buyers ":
        campaign_goal = "Encourage more frequent purchases"
        message = f"Still interested in {product}? Here is a limited-time personalized offer just for you."

    elif segment == "Inactive Customers":
        campaign_goal = "Win back inactive customers"
        message = f"We miss you! Come back and enjoy a special discount on {product}."

    else:
        campaign_goal = "General engagement"
        message = "Explore our latest offers selected specially for you."

    return pd.Series([campaign_goal, message])


def assign_segment_names(customer_features):
    cluster_summary = customer_features.groupby("Cluster").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean"
    })

    cluster_summary["Score"] = (
        cluster_summary["Recency"].rank(ascending=False) +
        cluster_summary["Frequency"].rank(ascending=True) +
        cluster_summary["Monetary"].rank(ascending=True)
    )

    sorted_clusters = cluster_summary.sort_values("Score", ascending=False).index.tolist()

    segment_names = [
        "Champions",
        "Loyal Customers",
        "Occasional Buyers",
        "Inactive Customers"
    ]

    segment_map = {}

    for i, cluster in enumerate(sorted_clusters):
        segment_map[cluster] = segment_names[i]

    customer_features["Segment"] = customer_features["Cluster"].map(segment_map)

    return customer_features


def run_customer_segmentation(df):
    required_columns = ["CustomerID", "InvoiceDate", "Description", "Quantity", "UnitPrice"]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.copy()

    # Clean missing values
    df = df.dropna(subset=["CustomerID"])
    df = df.dropna(subset=["Description"])

    # Convert date
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # Remove invalid purchase rows
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # Clean CustomerID format
    df["CustomerID"] = df["CustomerID"].astype(str).str.replace(".0", "", regex=False)

    # Create Total
    df["Total"] = df["Quantity"] * df["UnitPrice"]

    # Reference date
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # RFM calculation
    if "InvoiceNo" in df.columns:
        rfm = df.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Total", "sum")
        ).reset_index()
    else:
        rfm = df.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("CustomerID", "count"),
            Monetary=("Total", "sum")
        ).reset_index()

    # Top product purchased
    top_product = df.groupby(["CustomerID", "Description"])["Quantity"].sum().reset_index()

    top_product = top_product.sort_values(
        ["CustomerID", "Quantity"],
        ascending=[True, False]
    )

    top_product = top_product.drop_duplicates("CustomerID")
    top_product = top_product[["CustomerID", "Description"]]
    top_product = top_product.rename(columns={"Description": "Top_Product"})

    # Merge RFM and top product
    customer_features = rfm.merge(top_product, on="CustomerID", how="left")

    if len(customer_features) < 4:
        raise ValueError("At least 4 customers are needed for K-Means clustering.")

    # K-Means clustering
    X = customer_features[["Recency", "Frequency", "Monetary"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    customer_features["Cluster"] = kmeans.fit_predict(X_scaled)

    # Assign business segment names
    customer_features = assign_segment_names(customer_features)

    # Generate campaign message
    customer_features[["Campaign_Goal", "Campaign_Message"]] = customer_features.apply(
        generate_campaign,
        axis=1
    )

    return customer_features, df


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">AI-Driven Personalized Marketing Dashboard</div>',
    unsafe_allow_html=True
)


# ---------------- SIDEBAR ----------------
st.sidebar.title("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload customer transaction CSV",
    type=["csv"]
)

run_analysis = st.sidebar.button("Run Customer Analysis")


# ---------------- FILE UPLOAD ----------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="latin1")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        if run_analysis:
            with st.spinner("Analyzing customer behavior..."):
                customer_features, cleaned_df = run_customer_segmentation(df)

            st.session_state["customer_features"] = customer_features
            st.session_state["cleaned_df"] = cleaned_df


    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload a CSV file from the sidebar to begin analysis.")


# ---------------- DASHBOARD ----------------
if "customer_features" in st.session_state:

    customer_features = st.session_state["customer_features"]
    cleaned_df = st.session_state["cleaned_df"]

    total_customers = customer_features["CustomerID"].nunique()
    total_segments = customer_features["Segment"].nunique()
    total_revenue = customer_features["Monetary"].sum()
    avg_recency = customer_features["Recency"].mean()

    st.markdown("## 📌 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{total_customers}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Customer Segments</div>
            <div class="metric-value">{total_segments}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Revenue</div>
            <div class="metric-value">{total_revenue:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Recency</div>
            <div class="metric-value">{avg_recency:.1f} days</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "👥 Segment Analysis",
        "💬 Campaign Messages",
        "📄 Full Customer Data"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.subheader("Customer Segment Distribution")

        segment_counts = customer_features["Segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Customer_Count"]

        fig1 = px.pie(
            segment_counts,
            names="Segment",
            values="Customer_Count",
            hole=0.45,
            title="Customer Distribution by Segment"
        )

        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Segment Count Table")
        st.dataframe(segment_counts, use_container_width=True)

    # ---------------- TAB 2 ----------------
    with tab2:
        st.subheader("RFM Segment Summary")

        segment_summary = customer_features.groupby("Segment").agg({
            "CustomerID": "count",
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean"
        }).round(2)

        segment_summary = segment_summary.rename(columns={
            "CustomerID": "Customer Count",
            "Recency": "Avg Recency",
            "Frequency": "Avg Frequency",
            "Monetary": "Avg Monetary"
        })

        st.dataframe(segment_summary, use_container_width=True)

        summary_reset = segment_summary.reset_index()

        fig2 = px.bar(
            summary_reset,
            x="Segment",
            y="Avg Monetary",
            text="Avg Monetary",
            title="Average Monetary Value by Segment"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Customer Behaviour Map")

        fig3 = px.scatter(
            customer_features,
            x="Recency",
            y="Monetary",
            size="Frequency",
            color="Segment",
            hover_data=["CustomerID", "Top_Product"],
            title="Recency vs Monetary Value"
        )

        st.plotly_chart(fig3, use_container_width=True)

    # ---------------- TAB 3 ----------------
    with tab3:
        st.subheader("Personalized Campaign Output")

        final_output = customer_features[
            ["CustomerID", "Segment", "Top_Product", "Campaign_Goal", "Campaign_Message"]
        ]

        st.dataframe(final_output, use_container_width=True)

        csv = final_output.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Campaign CSV",
            data=csv,
            file_name="personalized_marketing_campaigns.csv",
            mime="text/csv"
        )

    # ---------------- TAB 4 ----------------
    with tab4:
        st.subheader("Full Customer Feature Table")

        st.dataframe(customer_features, use_container_width=True)

        full_csv = customer_features.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Full Customer Data",
            data=full_csv,
            file_name="customer_segments_full_data.csv",
            mime="text/csv"
        )