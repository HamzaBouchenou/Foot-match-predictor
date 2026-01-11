"""
Load and clean football data from CSV files
"""
import pandas as pd
import os
from glob import glob


def load_all_csv_files(data_path='../data/raw/'):
    """
    Load all CSV files from the raw data folder and combine them
    
    Args:
        data_path: Path to folder containing CSV files
        
    Returns:
        Combined pandas DataFrame
    """
    print("="*60)
    print("STEP 1: LOADING CSV FILES")
    print("="*60)
    
    # Get all CSV files in the folder
    csv_files = glob(os.path.join(data_path, '*.csv'))
    
    # Check if files exist
    if not csv_files:
        raise FileNotFoundError(f"❌ No CSV files found in {data_path}")
    
    print(f"✅ Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"   - {os.path.basename(file)}")
    
    # Load each CSV file
    all_dataframes = []
    
    for file in csv_files:
        print(f"\n📂 Loading {os.path.basename(file)}...")
        
        try:
            df = pd.read_csv(file, encoding='utf-8')
            print(f"   ✅ Loaded {len(df)} matches")
            all_dataframes.append(df)
            
        except Exception as e:
            print(f"   ❌ Error loading {file}: {e}")
            continue
    
    # Combine all dataframes into one
    if not all_dataframes:
        raise ValueError("❌ No data was loaded successfully!")
    
    print("\n" + "="*60)
    print("COMBINING ALL FILES...")
    print("="*60)
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"✅ Total matches loaded: {len(combined_df)}")
    print(f"✅ Total columns: {len(combined_df.columns)}")
    
    return combined_df


def select_needed_columns(df):
    """
    Keep only the columns we need for machine learning
    
    Args:
        df: Raw DataFrame with all columns
        
    Returns:
        DataFrame with only selected columns
    """
    print("\n" + "="*60)
    print("STEP 2: SELECTING NEEDED COLUMNS")
    print("="*60)
    
    # Define the columns we want to keep
    needed_columns = [
        'Date',           # Match date
        'HomeTeam',       # Home team name
        'AwayTeam',       # Away team name
        'FTHG',          # Full Time Home Goals
        'FTAG',          # Full Time Away Goals
        'FTR',           # Full Time Result (H/D/A) - TARGET VARIABLE
        'HS',            # Home Shots
        'AS',            # Away Shots
        'HST',           # Home Shots on Target
        'AST',           # Away Shots on Target
        'HC',            # 🆕 Home Corners
        'AC',            # 🆕 Away Corners
    ]
    
    print(f"📋 Columns we want: {len(needed_columns)}")
    for col in needed_columns:
        print(f"   - {col}")
    
    # Check which columns actually exist in the dataframe
    available_columns = [col for col in needed_columns if col in df.columns]
    missing_columns = [col for col in needed_columns if col not in df.columns]
    
    if missing_columns:
        print(f"\n⚠️  Warning: These columns are missing from the CSV:")
        for col in missing_columns:
            print(f"   - {col}")
    
    print(f"\n✅ Keeping {len(available_columns)} columns")
    
    # Select only available columns
    df_selected = df[available_columns].copy()
    
    print(f"✅ Data shape: {df_selected.shape[0]} rows × {df_selected.shape[1]} columns")
    
    return df_selected


def clean_data(df):
    """
    Clean the data: handle missing values, convert types, sort by date
    
    Args:
        df: DataFrame with selected columns
        
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*60)
    print("STEP 3: CLEANING DATA")
    print("="*60)
    
    print(f"📊 Starting with {len(df)} matches")
    
    # 3.1: Convert Date column to datetime
    print("\n🔧 Converting Date column to datetime format...")
    
    # Try different date formats
    date_formats = ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d']
    
    for date_format in date_formats:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format=date_format, errors='coerce')
            print(f"   ✅ Successfully parsed dates with format: {date_format}")
            break
        except:
            continue
    
    # If all formats failed, try automatic parsing
    if df['Date'].isna().all():
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # 3.2: Check for missing values in essential columns
    print("\n🔍 Checking for missing values in essential columns...")
    
    essential_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    
    before_cleaning = len(df)
    
    for col in essential_columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                print(f"   ⚠️  {col}: {missing_count} missing values")
    
    # Remove rows with missing essential data
    df = df.dropna(subset=[col for col in essential_columns if col in df.columns])
    
    after_cleaning = len(df)
    removed = before_cleaning - after_cleaning
    
    if removed > 0:
        print(f"   🗑️  Removed {removed} rows with missing essential data")
    else:
        print(f"   ✅ No missing essential data found")
    
    # 3.3: Fill missing shot statistics with 0 (if columns exist)
    shot_columns = ['HS', 'AS', 'HST', 'AST']
    
    for col in shot_columns:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"   🔧 Filling {missing} missing values in {col} with 0")
                df[col] = df[col].fillna(0)
    
    # 3.4: Convert goal columns to integers
    print("\n🔢 Converting goal columns to numeric format...")
    
    for col in ['FTHG', 'FTAG']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # 3.5: Sort by date (MOST IMPORTANT!)
    print("\n📅 Sorting matches by date (oldest to newest)...")
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"   ✅ First match: {df['Date'].min()}")
    print(f"   ✅ Last match: {df['Date'].max()}")

    print("\n🔖 Adding Match IDs...")
    df.insert(0, 'match_id', ['match_' + str(i).zfill(4) for i in range(1, len(df) + 1)])
    print(f"   ✅ Added {len(df)} unique match IDs")

    print("\n🏆 Adding League information...")
    df.insert(1, 'League', 'Premier League')  # Insert after match_id
    print(f"   ✅ Set league: Premier League for all {len(df)} matches")

    # 3.6: Show final statistics
    print("\n" + "="*60)
    print("CLEANING COMPLETE!")
    print("="*60)
    print(f"✅ Final dataset: {len(df)} matches")
    print(f"✅ Columns: {list(df.columns)}")
    print(f"✅ Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Show data types
    print("\n📋 Data types:")
    print(df.dtypes)
    
    return df


def save_processed_data(df, output_path='../data/processed/final_dataset.csv'):
    """
    Save the cleaned data to CSV
    
    Args:
        df: Cleaned DataFrame
        output_path: Path where to save the file
    """
    print("\n" + "="*60)
    print("STEP 4: SAVING PROCESSED DATA")
    print("="*60)
    
    # Create the processed folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✅ Data saved to: {output_path}")
    print(f"✅ File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def show_data_preview(df, n_rows=5):
    """
    Show a preview of the data
    
    Args:
        df: DataFrame to preview
        n_rows: Number of rows to show
    """
    print("\n" + "="*60)
    print("DATA PREVIEW")
    print("="*60)
    
    print(f"\n📊 First {n_rows} matches:")
    print(df.head(n_rows).to_string())
    
    print(f"\n📊 Last {n_rows} matches:")
    print(df.tail(n_rows).to_string())
    
    # Show outcome distribution
    if 'FTR' in df.columns:
        print("\n📈 Match outcomes distribution:")
        outcome_counts = df['FTR'].value_counts()
        total = len(df)
        
        for outcome, count in outcome_counts.items():
            percentage = (count / total) * 100
            outcome_name = {'H': 'Home Wins', 'D': 'Draws', 'A': 'Away Wins'}.get(outcome, outcome)
            print(f"   {outcome_name}: {count} ({percentage:.1f}%)")


# Main execution function
def process_football_data(input_path='../data/raw/', output_path='../data/processed/final_dataset.csv'):
    """
    Complete pipeline: Load → Select → Clean → Save
    
    Args:
        input_path: Folder containing raw CSV files
        output_path: Where to save cleaned data
        
    Returns:
        Cleaned DataFrame
    """
    try:
        # Step 1: Load all CSV files
        df_raw = load_all_csv_files(input_path)
        
        # Step 2: Select needed columns
        df_selected = select_needed_columns(df_raw)
        
        # Step 3: Clean the data
        df_clean = clean_data(df_selected)
        
        # Step 4: Save processed data
        save_processed_data(df_clean, output_path)
        
        # Step 5: Show preview
        show_data_preview(df_clean)
        
        print("\n" + "="*60)
        print("✅ ALL DONE! DATA IS READY FOR FEATURE ENGINEERING")
        print("="*60)
        
        return df_clean
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


# This runs when you execute: python src/data_loader.py
if __name__ == "__main__":
    df = process_football_data()