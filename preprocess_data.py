import pandas as pd
import numpy as np
from datetime import datetime

# VEHICLE DATA PREPROCESSING SCRIPT

def load_data(filepath):
    """Load the raw CSV file"""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    print(f" Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    return df


def remove_duplicates(df):
    """Remove duplicate rows based on URL (same ad listed multiple times)"""
    before = len(df)
    df = df.drop_duplicates(subset=['url'], keep='first')
    after = len(df)
    print(f" Removed {before - after} duplicate rows → {after} remaining")
    return df.reset_index(drop=True)


def drop_unnecessary_columns(df):
    """Drop columns not useful for ML modeling"""
    cols_to_drop = ['title', 'details', 'contact', 'url', 'scrape_date']
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    print(f" Dropped columns: {existing}")
    print(f" Remaining columns: {list(df.columns)}")
    return df


def clean_mileage(df):
    """Convert mileage to numeric, handle '-' and missing values"""
    # Replace '-' and empty values with NaN
    df['mileage'] = df['mileage'].replace(['-', '', ' '], np.nan)
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    
    missing = df['mileage'].isna().sum()
    print(f" Mileage: {missing} missing values")
    
    # Flag unrealistic mileage values (> 500,000 km likely data entry errors)
    unrealistic = (df['mileage'] > 500000).sum()
    if unrealistic > 0:
        print(f"   {unrealistic} rows with mileage > 500,000 km → set to NaN")
        df.loc[df['mileage'] > 500000, 'mileage'] = np.nan
    
    # Flag very low mileage (< 10 km for non-brand-new cars)
    # Only flag if car is older than 1 year
    current_year = datetime.now().year
    suspicious_low = ((df['mileage'] < 10) & (df['yom'] < current_year - 1)).sum()
    if suspicious_low > 0:
        print(f"   {suspicious_low} old cars with mileage < 10 km → set to NaN")
        df.loc[(df['mileage'] < 10) & (df['yom'] < current_year - 1), 'mileage'] = np.nan
    
    return df


def clean_engine_cc(df):
    """Convert engine_cc to numeric, handle '-' and invalid values"""
    df['engine_cc'] = df['engine_cc'].replace(['-', '', ' '], np.nan)
    df['engine_cc'] = pd.to_numeric(df['engine_cc'], errors='coerce')
    
    missing = df['engine_cc'].isna().sum()
    print(f" Engine CC: {missing} missing values")
    
    # Flag unrealistic engine sizes (< 100 cc or > 5000 cc for cars)
    unrealistic_low = ((df['engine_cc'] < 100) & df['engine_cc'].notna()).sum()
    unrealistic_high = ((df['engine_cc'] > 5000) & df['engine_cc'].notna()).sum()
    
    if unrealistic_low > 0:
        print(f"   {unrealistic_low} rows with engine_cc < 100 → set to NaN")
        df.loc[df['engine_cc'] < 100, 'engine_cc'] = np.nan
    if unrealistic_high > 0:
        print(f"   {unrealistic_high} rows with engine_cc > 5000 → set to NaN")
        df.loc[df['engine_cc'] > 5000, 'engine_cc'] = np.nan
    
    return df


def clean_price(df):
    """Handle price outliers"""
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Remove rows with no price
    before = len(df)
    df = df.dropna(subset=['price'])
    print(f" Price: removed {before - len(df)} rows with no price")
    
    # Remove extreme outliers (> 100M is almost certainly data entry error)
    extreme = (df['price'] > 100_000_000).sum()
    if extreme > 0:
        print(f"   Removing {extreme} rows with price > 100,000,000 (data entry errors)")
        df = df[df['price'] <= 100_000_000]
    
    # Remove suspiciously low prices (< 50,000 Rs - likely partial/incorrect)
    low = (df['price'] < 50_000).sum()
    if low > 0:
        print(f"   Removing {low} rows with price < 50,000 (likely errors)")
        df = df[df['price'] >= 50_000]
    
    # Flag remaining outliers for info
    q1 = df['price'].quantile(0.01)
    q99 = df['price'].quantile(0.99)
    outliers = ((df['price'] < q1) | (df['price'] > q99)).sum()
    if outliers > 0:
        print(f"   {outliers} price outliers remain (below {q1:,.0f} or above {q99:,.0f})")
    
    return df.reset_index(drop=True)


def clean_yom(df):
    """Clean Year of Manufacture"""
    df['yom'] = pd.to_numeric(df['yom'], errors='coerce')
    
    current_year = datetime.now().year
    
    # Remove rows with no year
    before = len(df)
    df = df.dropna(subset=['yom'])
    print(f" YOM: removed {before - len(df)} rows with no year")
    
    # Remove unrealistic years (yom=0, yom < 1950, or yom > current_year + 1)
    unrealistic = ((df['yom'] < 1950) | (df['yom'] > current_year + 1)).sum()
    if unrealistic > 0:
        print(f"   Removing {unrealistic} rows with year < 1950 or > {current_year + 1}")
        df = df[(df['yom'] >= 1950) & (df['yom'] <= current_year + 1)]
    
    # Convert to int
    df['yom'] = df['yom'].astype(int)
    
    return df.reset_index(drop=True)


def clean_location(df):
    """Handle 'Unknown' locations"""
    unknown_count = (df['location'] == 'Unknown').sum()
    print(f" Location: {unknown_count} 'Unknown' values")
    
    # Strip whitespace
    df['location'] = df['location'].str.strip()
    
    return df


def clean_options(df):
    """Clean options field and handle '-' values"""
    df['options'] = df['options'].replace(['-', '', ' '], 'None')
    df['options'] = df['options'].fillna('None')
    
    # Standardize option text (uppercase, consistent separators)
    df['options'] = df['options'].str.upper().str.strip()
    
    # Replace inconsistent separators
    df['options'] = df['options'].str.replace(r'\s*,\s*', ', ', regex=True)
    
    return df


def clean_categorical(df):
    """Standardize categorical columns"""
    # Standardize make names
    df['make'] = df['make'].str.strip().str.title()
    
    # Fix known make inconsistencies
    make_fixes = {
        'Mercedes-Benz': 'Mercedes-Benz',
        'Mercedes': 'Mercedes-Benz',
        'Mg': 'MG',
        'Bmw': 'BMW',
    }
    df['make'] = df['make'].replace(make_fixes)
    
    # Standardize model names
    df['model'] = df['model'].str.strip()
    
    # Standardize gear - fix invalid values from scraper parsing errors
    df['gear'] = df['gear'].str.strip().str.capitalize()
    valid_gears = ['Automatic', 'Manual']
    invalid_gears = df[~df['gear'].isin(valid_gears)]['gear'].unique()
    if len(invalid_gears) > 0:
        print(f"   Invalid gear values found: {list(invalid_gears)} → set to 'Unknown'")
        df.loc[~df['gear'].isin(valid_gears), 'gear'] = 'Unknown'
    
    # Standardize fuel_type - fix invalid values
    df['fuel_type'] = df['fuel_type'].str.strip().str.capitalize()
    valid_fuels = ['Petrol', 'Diesel', 'Hybrid', 'Electric']
    invalid_fuels = df[~df['fuel_type'].isin(valid_fuels)]['fuel_type'].unique()
    if len(invalid_fuels) > 0:
        print(f"   Invalid fuel_type values found: {list(invalid_fuels)} → set to 'Unknown'")
        df.loc[~df['fuel_type'].isin(valid_fuels), 'fuel_type'] = 'Unknown'
    
    print(f" Standardized categorical columns")
    print(f"   Makes: {sorted(df['make'].unique())}")
    print(f"   Gears: {sorted(df['gear'].unique())}")
    print(f"   Fuel types: {sorted(df['fuel_type'].unique())}")
    
    return df


def fill_missing_values(df):
    """Fill remaining missing values with appropriate strategies"""
    # Fill mileage with median (grouped by make if possible)
    if df['mileage'].isna().any():
        median_mileage = df['mileage'].median()
        df['mileage'] = df.groupby('make')['mileage'].transform(
            lambda x: x.fillna(x.median() if x.median() is not np.nan else median_mileage)
        )
        # If still NaN (make had no valid mileage), fill with overall median
        df['mileage'] = df['mileage'].fillna(median_mileage)
        print(f" Filled missing mileage with group median ({median_mileage:,.0f} km overall)")
    
    # Fill engine_cc with median per make
    if df['engine_cc'].isna().any():
        median_cc = df['engine_cc'].median()
        df['engine_cc'] = df.groupby('make')['engine_cc'].transform(
            lambda x: x.fillna(x.median() if x.median() is not np.nan else median_cc)
        )
        df['engine_cc'] = df['engine_cc'].fillna(median_cc)
        print(f" Filled missing engine_cc with group median ({median_cc:.0f} cc overall)")
    
    return df


def print_summary(df):
    """Print a summary of the cleaned dataset"""
    print("\n" + "=" * 60)
    print(" PREPROCESSED DATA SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\n--- Numeric Columns ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        print(f"  {col:15s}: min={df[col].min():>12,.0f}  max={df[col].max():>12,.0f}  "
              f"mean={df[col].mean():>12,.0f}  missing={df[col].isna().sum()}")
    
    print(f"\n--- Categorical Columns ---")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        unique = df[col].nunique()
        print(f"  {col:15s}: {unique} unique values")
    
    print(f"\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print("  None! ")
    print("=" * 60)


# MAIN PREPROCESSING PIPELINE

if __name__ == "__main__":
    print("=" * 60)
    print(" VEHICLE DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    filepath = input("\nEnter CSV file path: ").strip()
    if not filepath:
        # Default: find the most recent CSV
        import glob
        csvs = glob.glob("riyasewana_search_*.csv")
        if csvs:
            filepath = sorted(csvs)[-1]  # Most recent
            print(f"   Using most recent: {filepath}")
        else:
            print(" No CSV file found!")
            exit(1)
    
    df = load_data(filepath)
    
    print(f"\n{'─' * 60}")
    print("STEP 1: Remove Duplicates")
    print(f"{'─' * 60}")
    df = remove_duplicates(df)
    
    print(f"\n{'─' * 60}")
    print("STEP 2: Clean Price")
    print(f"{'─' * 60}")
    df = clean_price(df)
    
    print(f"\n{'─' * 60}")
    print("STEP 3: Clean Year of Manufacture")
    print(f"{'─' * 60}")
    df = clean_yom(df)
    
    print(f"\n{'─' * 60}")
    print("STEP 4: Clean Mileage")
    print(f"{'─' * 60}")
    df = clean_mileage(df)
    
    print(f"\n{'─' * 60}")
    print("STEP 5: Clean Engine CC")
    print(f"{'─' * 60}")
    df = clean_engine_cc(df)
    
    print(f"\n{'─' * 60}")
    print("STEP 6: Clean Location")
    print(f"{'─' * 60}")
    df = clean_location(df)
    
    print(f"\n{'─' * 60}")
    print("STEP 7: Clean Options")
    print(f"{'─' * 60}")
    df = clean_options(df)
    
    print(f"\n {'─' * 60}")
    print("STEP 8: Standardize Categorical Columns")
    print(f"{'─' * 60}")
    df = clean_categorical(df)
    
    print(f"\n{'─' * 60}")
    print("STEP 9: Fill Missing Values")
    print(f"{'─' * 60}")
    df = fill_missing_values(df)
    
    print(f"\n{'─' * 60}")
    print("STEP 10: Drop Unnecessary Columns")
    print(f"{'─' * 60}")
    df = drop_unnecessary_columns(df)
    
    # Summary
    print_summary(df)
    
    # Save
    output_file = filepath.replace('.csv', '_preprocessed.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n Saved preprocessed data to: {output_file}")
    print(f"   {len(df)} rows × {len(df.columns)} columns")
    
    # Show a sample
    print(f"\n Sample (first 5 rows):")
    print(df.head().to_string())
