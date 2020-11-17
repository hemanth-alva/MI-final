import pandas as pd
def DataPreprocessing(data):
    ''' Data PreProcessing 
    - Filling of Missing Values
    '''

    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Delivery phase'] = data['Delivery phase'].fillna(data['Delivery phase'].median())
    data['Education'] = data['Education'].fillna(data['Education'].mode()[0])
    data['Residence'] = data['Residence'].fillna(0)
    data['BP'] = data['BP'].fillna(data['BP'].dropna().mean())
    data['HB'] = data['HB'].fillna(data['HB'].dropna().median())
    data['Weight'] = data['Weight'].fillna(data['Weight'].dropna().median())
    return data

data = pd.read_csv("C:/Users/Sneha Jayaraman/Downloads/LBW_Dataset.csv")
data = DataPreprocessing(data)
data.to_csv('CleanedLBW_Dataset.csv', index=False)
