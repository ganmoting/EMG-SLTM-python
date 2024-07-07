import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
torch.manual_seed(50)
np.random.seed(50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# 自定义数据集类
class EMGDataset(Dataset):
    def __init__(self, csv_file, label):
        self.data = np.genfromtxt(csv_file, delimiter=',')
        self.label = label
        self.cycle_length = 10000

    def __len__(self):
        return len(self.data) // self.cycle_length

    def __getitem__(self, idx):
        start = idx * self.cycle_length
        end = start + self.cycle_length
        cycle = self.data[start:end].reshape(-1, 1)  # Shape: (10000, 1)
        return torch.FloatTensor(cycle), torch.FloatTensor([self.label])



# 1. 加载数据
handshake_dataset = EMGDataset(r'E:\demo\data\strain\Sheet1_1.csv', 0)
raise_hand_dataset = EMGDataset(r'E:\demo\data\strain\Sheet2_1.csv', 1)

# 合并数据集
full_dataset = torch.utils.data.ConcatDataset([handshake_dataset, raise_hand_dataset])

# 分割数据集
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义LSTM模型
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


# 实例化模型、损失函数和优化器
model = EMGClassifier(input_size=1, hidden_size=64, num_layers=3, output_size=1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y.squeeze())
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

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        predicted = (outputs.squeeze() > 0.5).float()
        total += batch_y.size(0)
        correct += (predicted == batch_y.squeeze()).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')


# 使用模型进行预测
def predict_action(emg_data):
    if emg_data.shape != (10000, 1):
        raise ValueError("Input data should have shape (10000, 1)")

    emg_tensor = torch.FloatTensor(emg_data).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(emg_tensor)

    action = "握手" if output.item() < 0.5 else "举手"
    confidence = output.item() if output.item() > 0.5 else 1 - output.item()

    return action, confidence


# 示例：预测新的EMG数据
new_emg_data = np.random.rand(10000, 1)  # 假设我们有一个新的动作周期的EMG数据
action, confidence = predict_action(new_emg_data)
print(f"预测动作: {action}, 置信度: {confidence:.2f}")