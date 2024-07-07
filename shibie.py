import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import flask

from train import LSTMModel

# 加载模型
try:
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=1)
    model.load_state_dict(torch.load(r'data\lstm_model.pth', map_location=torch.device('cpu')))
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# 加载数据进行预测
file_path = r'E:\demo\data\strain\Sheet2.csv'  # 替换为你的预测数据文件路径
try:
    data = np.genfromtxt(file_path, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(-1, 1)
except Exception as e:
    print(f"Error loading data from {file_path}: {e}")
    raise

# 假设数据预处理与训练时相同
scaler = StandardScaler()  # 使用与训练时相同的标准化器
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
X_new = torch.FloatTensor(data_scaled).unsqueeze(0)

# 使用模型进行预测
model.eval()
with torch.no_grad():
    outputs = model(X_new)
    predictions = outputs.squeeze().numpy()
    print(predictions)
