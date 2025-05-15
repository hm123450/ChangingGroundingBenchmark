import pandas as pd


df = pd.read_csv('choose4/today250_updated1.csv')


def string_to_list(s):

    return [word for word in s.split()]


df['tokens'] = df.iloc[:, 3].apply(string_to_list)


df.to_csv('choose4/today250_updated2.csv', index=False)
