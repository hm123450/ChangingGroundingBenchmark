import pandas as pd

# 读取CSV文件
df = pd.read_csv('choose4/today250_updated1.csv')

# 定义一个函数，将字符串转换为列表
def string_to_list(s):
    # 将字符串分割成单词列表
    return [word for word in s.split()]

# 应用这个函数到第二列，创建新列'token'
df['tokens'] = df.iloc[:, 3].apply(string_to_list)

# 将修改后的DataFrame写回到新的CSV文件
df.to_csv('choose4/today250_updated2.csv', index=False)