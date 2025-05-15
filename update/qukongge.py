import pandas as pd
import re
# 读取CSV文件
df = pd.read_csv('choose4/today250_updated.csv')  # 替换'your_file.csv'为你的文件路径
def remove_pattern(text):
    # 使用正则表达式替换一个或多个空格后跟'%anchor_article%'为''
    return re.sub(r'\s{2,}', ' ', text)
# 假设我们要处理的列名为'ColumnName'
column_name = 'utterance'  # 替换'ColumnName'为你的列名
print(df[column_name])
# 遍历列中的每个元素，检查并删除'%anchor_article%'
df[column_name] = df[column_name].apply(lambda x: remove_pattern(x) if '  ' in x else x)

# 将更新后的DataFrame保存回CSV文件
df.to_csv('choose4/today250_updated1.csv', index=False)  # 替换'updated_file.csv'为你想要保存的新文件路径