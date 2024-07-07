import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 1. 加载数据
handshake_data = np.genfromtxt(r'E:\demo\data\strain\Sheet1_1.csv', delimiter=',')
raise_hand_data = np.genfromtxt(r'E:\demo\data\strain\Sheet2_1.csv', delimiter=',')

# 确保数据是2D的
if handshake_data.ndim == 1:
    handshake_data = handshake_data.reshape(-1, 1)
if raise_hand_data.ndim == 1:
    raise_hand_data = raise_hand_data.reshape(-1, 1)

print(f"Handshake data shape: {handshake_data.shape}")
print(f"Raise hand data shape: {raise_hand_data.shape}")

# 2. 数据预处理和标签创建
# 创建标签（0为握手，1为举手）
handshake_labels = np.zeros(len(handshake_data))
raise_hand_labels = np.ones(len(raise_hand_data))
#
# 合并数据和标签
X = np.vstack((handshake_data, raise_hand_data))
y = np.hstack((handshake_labels, raise_hand_labels))

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(2).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(2).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 3. 定义LSTM模型
class EMGClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EMGClassifier, self).__init__()
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
model = EMGClassifier(input_size=1, hidden_size=64, num_layers=2, output_size=1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 训练模型
num_epochs = 100
train_losses = []

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

# 保存模型
torch.save(model.state_dict(), 'emg_classifier.pth')

# 6. 评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs.squeeze(), y_test_tensor)
    test_accuracy = ((test_outputs.squeeze() > 0.5) == y_test_tensor).float().mean()
    print(f'Test Loss: {test_loss.item():.6f}, Test Accuracy: {test_accuracy.item():.6f}')


# 7. 使用模型进行预测
def predict_action(emg_data):
    if emg_data.ndim == 1:
        emg_data = emg_data.reshape(-1, 1)

    emg_data_scaled = scaler.transform(emg_data)
    emg_tensor = torch.FloatTensor(emg_data_scaled).unsqueeze(0).unsqueeze(2).to(device)

    model.eval()
    with torch.no_grad():
        output = model(emg_tensor)

    action = "握手" if output.item() < 0.5 else "举手"
    confidence = output.item() if output.item() > 0.5 else 1 - output.item()

    return action, confidence


# 示例：预测新的EMG数据
new_emg_data = np.random.rand(100, 1)  # 假设我们有100个时间点的新EMG数据
action, confidence = predict_action(new_emg_data)
print(f"预测动作: {action}, 置信度: {confidence:.2f}")