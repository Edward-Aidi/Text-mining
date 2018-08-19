#READING FORD DATA

#import pandas package
import pandas as pd

# Assign spreadsheet filename to 'file'
file = '/Users/nevskaya/Dropbox/TextAnalytics/DATASETS/Ford_data/Ford_data.xlsx'
xl = pd.ExcelFile(file)     # Load spreadsheet
print(xl.sheet_names)       # Print the sheet names
df = xl.parse('Ford_data')  # Load a sheet into a DataFrame by name: df
df.head()                   # See first lines of the dataframe

# Describe data: value for class variable and counts?
df['class'].value_counts().plot.bar();

#HOW CAN YOU USE THE postive and negative classes?