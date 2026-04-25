import pandas as pd
import os

# ==========================================
# 1. Setup Input and Output Paths
# ==========================================
# Update this path to where your actual 35-sheet excel file is located
input_excel_path = r"/Users/canberkkurtul/Desktop/emgdata/training_2304/zeynep_data_2304.xlsx"

# Where you want the final combined file to be saved
output_file_path = r"/Users/canberkkurtul/Desktop/new_training_2304/zeynep_combined_2304.csv"

# ==========================================
# 2. Define the Classification Logic
# ==========================================
def get_class_label(sheet_name):
    # Convert to lowercase to avoid capitalization issues
    name = sheet_name.lower()
    
    if 'right biceps' in name:
        return 0
    elif 'right triceps' in name:
        return 1
    elif 'right frontarm' in name:
        return 2
    elif 'left biceps' in name:
        return 3
    elif 'left triceps' in name or 'letf triceps' in name: # Catches the "letf" typo!
        return 4
    elif 'left frontarm' in name:
        return 5
    elif 'rest' in name:
        return 6
    else:
        print(f"Warning: Could not assign a class for sheet named: '{sheet_name}'")
        return None

# ==========================================
# 3. Execution
# ==========================================
def combine_emg_data():
    print("Reading Excel file... (This might take a few seconds)")
    
    # Setting sheet_name=None tells pandas to load ALL sheets into a dictionary
    all_sheets = pd.read_excel(input_excel_path, sheet_name=None)
    
    combined_data_list = []
    
    for sheet_name, df in all_sheets.items():
        label = get_class_label(sheet_name)
        
        if label is not None:
            # Add the TRUECLASS column at the very end
            df['TRUECLASS'] = label
            
            # Store the processed dataframe in our list
            combined_data_list.append(df)
            print(f"Processed: {sheet_name: <20} | Added TRUECLASS: {label} | Rows: {len(df)}")

    # Bind all the individual dataframes together into one giant matrix
    final_master_df = pd.concat(combined_data_list, ignore_index=True)
    
    # Save the combined dataset to a new CSV file
    final_master_df.to_csv(output_file_path, index=False)
    
    # Note: If you want it saved as an Excel file instead of CSV, comment out the line above 
    # and uncomment the line below:
    # final_master_df.to_excel(output_file_path.replace('.csv', '.xlsx'), index=False)

    print("\n" + "="*60)
    print("SUCCESS! Data Binding Complete.")
    print(f"Total Rows in Final Dataset: {len(final_master_df)}")
    print(f"Total Columns (24 features + 1 Trueclass): {len(final_master_df.columns)}")
    print(f"Saved to: {output_file_path}")
    print("="*60)

if __name__ == "__main__":
    combine_emg_data()