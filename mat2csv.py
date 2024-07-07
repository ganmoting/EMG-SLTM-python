import scipy.io
import numpy as np
import pandas as pd

# 读取MATLAB文件
mat = scipy.io.loadmat('test\matlab1.mat')

# 提取数据（假设数据在变量 data 中）
data = mat['EMG']

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 保存为CSV文件
df.to_csv('jidian1.csv', index=False, float_format="%.2e")

print("CSV 文件已保存成功。")
