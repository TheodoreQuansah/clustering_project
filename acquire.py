import pandas as pd


def wine_df():
    # Read data from two separate CSV files for white and red wines
    df_1 = pd.read_csv('winequality_white.csv')
    df_2 = pd.read_csv('winequality_red.csv')
    
    # Add a 'wine_type' column to distinguish between red and white wines
    df_1['wine_type'] = 'white'
    df_2['wine_type'] = 'red'
    
    # Concatenate the DataFrames vertically to combine white and red wine data
    df = pd.concat([df_1, df_2], axis=0)
    
    # Reset the index if needed (optional)
    df.reset_index(drop=True, inplace=True)
    
    # Save the combined DataFrame to a new CSV file named 'wine.csv'
    df.to_csv("wine.csv", index=False)

    # Return the combined DataFrame
    return df

