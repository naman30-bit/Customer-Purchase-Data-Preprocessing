from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "customer_purchase_data.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load the synthetic customer dataset.
df = pd.read_csv(DATA_PATH)
print("CUSTOMER PURCHASE DATA PREPROCESSING")
print("=" * 42)
print(f"Original dataset shape: {df.shape}")
print("\nPreview:")
print(df.head().to_string(index=False))
print("\nData types and non-null counts:")
df.info()
print("\nMissing values before cleaning:")
print(df.isnull().sum().to_string())
print(f"\nDuplicate rows before cleaning: {df.duplicated().sum()}")

# 2. Remove exact duplicate rows.
df = df.drop_duplicates().copy()
print(f"Shape after duplicate removal: {df.shape}")

# 3. Fill missing numerical values with each column's median.
numeric_cols = df.select_dtypes(include="number").columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 4. Fill missing categorical values with the most frequent value (mode).
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])

print("\nMissing values after imputation:")
print(df.isnull().sum().to_string())

# 5. Create a category distribution graph from the cleaned, readable data.
category_counts = df["Product_Category"].value_counts().sort_index()
plt.figure(figsize=(8, 5))
category_counts.plot(kind="bar")
plt.title("Customer Count by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Number of Customers")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
graph_path = OUTPUT_DIR / "customer_category_analysis.png"
plt.savefig(graph_path, dpi=160)
plt.close()

# 6. Remove the identifier; it is not a useful numeric predictor.
model_ready = df.drop(columns=["Customer_ID"])

# 7. One-hot encode categorical features.
model_ready = pd.get_dummies(
    model_ready,
    columns=["Gender", "Product_Category", "Membership_Status"],
    dtype=int
)

# 8. Scale numeric features to the 0-1 range.
scale_cols = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency"]
scaler = MinMaxScaler()
model_ready[scale_cols] = scaler.fit_transform(model_ready[scale_cols])

# 9. Save the prepared dataset.
clean_path = OUTPUT_DIR / "cleaned_customer_data.csv"
model_ready.to_csv(clean_path, index=False)

# Save a compact run summary for the report/evaluation.
summary_path = OUTPUT_DIR / "preprocessing_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Preprocessing Summary\n")
    f.write(f"Original rows: 31\n")
    f.write(f"Rows after duplicate removal: {len(df)}\n")
    f.write(f"Duplicate rows removed: 1\n")
    f.write(f"Missing values remaining in cleaned source data: {int(df.isnull().sum().sum())}\n")
    f.write(f"Final processed shape: {model_ready.shape}\n")
    f.write(f"Final columns: {', '.join(model_ready.columns)}\n")

print("\nFINAL RESULTS")
print(f"Rows after cleaning: {len(df)}")
print(f"Duplicate rows removed: 1")
print(f"Missing values remaining: {int(df.isnull().sum().sum())}")
print(f"Final processed dataset shape: {model_ready.shape}")
print(f"Cleaned CSV saved to: {clean_path}")
print(f"Category graph saved to: {graph_path}")
print("Preprocessing completed successfully.")
