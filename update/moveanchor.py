import pandas as pd
import re

df = pd.read_csv('choose4/today250.csv')  
def remove_pattern(text):

    return re.sub(r'\s+%anchor_article%', '', text)

column_name = 'utterance' 
print(df[column_name])

df[column_name] = df[column_name].apply(lambda x: remove_pattern(x) if '%anchor_article%' in x else x)


df.to_csv('choose4/today250_updated.csv', index=False)  
