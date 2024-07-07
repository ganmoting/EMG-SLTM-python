import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
torch.manual_seed(50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 1. 加载数据
file_path1 = r'E:\demo\data\strain\Sheet1_1.csv'
file_path2 = r'E:\demo\data\strain\Sheet2_1.csv'

try:
    data1 = np.genfromtxt(file_path1, delimiter=',')
    if data1.ndim == 1:
        data1 = data1.reshape(-1, 1)

    data2 = np.genfromtxt(file_path2, delimiter=',')
    if data2.ndim == 1:
        data2 = data2.reshape(-1, 1)
except Exception as e:
    print(f"Error loading data: {e}")
    raise

print(f"Data1 shape: {data1.shape}")
print(f"Data2 shape: {data2.shape}")

# 2. 数据预处理和标签创建
# 重塑数据为 (n_samples, time_steps, n_features)
n_samples1 = len(data1) // 10000
n_samples2 = len(data2) // 10000

X1 = data1.reshape((n_samples1, 10000, 1))
X2 = data2.reshape((n_samples2, 10000, 1))

# 创建标签（假设 data1 是类别 0，data2 是类别 1）
y1 = np.zeros(n_samples1)
y2 = np.ones(n_samples2)

# 合并数据和标签
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 3. 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


# 4. 实例化模型、损失函数和优化器
model = LSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=1).to(device)
criterion = nn.BCELoss()
#optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)

# 记录训练过程中的损失
train_losses = []

# 5. 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.6f}')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

torch.save(model.state_dict(), r'data\lstm_model.pth')

# 6. 评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs.squeeze(), y_test_tensor)
    test_accuracy = ((test_outputs.squeeze() > 0.5) == y_test_tensor).float().mean()
    print(f'Test Loss: {test_loss.item():.6f}, Test Accuracy: {test_accuracy.item():.6f}')

# 7. 使用模型进行预测
predictions = model(X_test_tensor).squeeze().detach().cpu().numpy()
print(predictions)

