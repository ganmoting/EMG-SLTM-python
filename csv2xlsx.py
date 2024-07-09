import pandas as pd

# 读取CSV文件
csv_file = 'data/strain/sheet2_3.csv'

# 加载CSV文件数据
data = pd.read_csv(csv_file)

# 定义Excel文件名
xlsx_file = 'test/output_file1.xlsx'

# 将数据保存为Excel文件
data.to_excel(xlsx_file, index=False)

print(f"CSV 文件 '{csv_file}' 已保存为 Excel 文件 '{xlsx_file}'。")
