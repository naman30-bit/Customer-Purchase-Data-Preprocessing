# Customer Purchase Data Analysis and Preprocessing

**Internship task:** Yuvaintern — Week 1 (Python for Machine Learning & Data Preprocessing)  
**Level:** Beginner-friendly B.Tech 2nd-year AIML  
**Dataset:** Synthetic sample data created only for learning; it contains no real customer information.

## Project structure
```text
Customer_Data_Preprocessing/
├── dataset/
│   └── customer_purchase_data.csv
├── output/
│   ├── cleaned_customer_data.csv       # generated when the script runs
│   ├── customer_category_analysis.png  # generated when the script runs
│   └── preprocessing_summary.txt       # generated when the script runs
├── preprocess.py
├── requirements.txt
├── README.md
└── Week1_Customer_Preprocessing_Report.docx
```

## Requirements
- Python 3.10 or newer recommended
- VS Code with the Python extension
- Internet connection for the first-time package installation

## Run in VS Code (Windows)
1. Extract the ZIP file.
2. Open the `Customer_Data_Preprocessing` folder in VS Code (`File > Open Folder`).
3. Open Terminal > New Terminal.
4. Check Python: `python --version`
5. (Recommended) Create a virtual environment:
   `python -m venv .venv`
6. Activate it in PowerShell:
   `.\.venv\Scripts\Activate.ps1`
   If PowerShell blocks activation, use Command Prompt and run:
   `.venv\Scripts\activate.bat`
7. Install libraries:
   `python -m pip install -r requirements.txt`
8. Run:
   `python preprocess.py`

If the `python` command is not recognized on your PC, try `py` instead (for example, `py --version` and `py preprocess.py`).

## What the script demonstrates
1. Loading a CSV with Pandas.
2. Inspecting shape, preview, data types, null values and duplicates.
3. Removing exact duplicate rows.
4. Imputing numerical missing values with the median and categorical missing values with the mode.
5. Removing `Customer_ID`, which is an identifier rather than a useful customer measurement.
6. One-hot encoding `Gender`, `Product_Category`, and `Membership_Status`.
7. Scaling numerical features with `MinMaxScaler` into the 0–1 range.
8. Creating a product-category bar chart.
9. Saving the processed CSV and a text summary in `output/`.

## Expected checks
The supplied raw CSV has 31 rows: 30 distinct customer records plus one exact duplicate. The script should remove one duplicate and leave 30 rows. The processed output has 30 rows and 13 columns (four scaled numeric columns plus nine one-hot columns). Verify these results by running the script on your computer.

## Important notes
- This is a preprocessing demonstration, not a trained prediction model.
- The scaler is fitted to this full sample dataset for demonstration. In a real predictive ML workflow, split data first and fit preprocessing steps only on training data to avoid data leakage.
- Do not describe the sample as real customer data.
- Add your own name, institute details and genuine screenshots before submitting the report.
