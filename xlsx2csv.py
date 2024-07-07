import pandas as pd

# 读取Excel文件
excel_file = 'test\output_file1.xlsx'

# 加载Excel文件中的所有工作表
excel_data = pd.read_excel(excel_file, sheet_name=None)

# 遍历所有工作表并保存为CSV文件
for sheet_name, data in excel_data.items():
    # 创建CSV文件名（使用工作表名）
    csv_file = f'{sheet_name}.csv'

    # 将数据保存为CSV文件
    data.to_csv(csv_file, index=False)

    print(f"工作表 {sheet_name} 已保存为 CSV 文件 {csv_file}。")
