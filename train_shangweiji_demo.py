import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import serial
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
                             QComboBox, QLabel, QWidget, QTextEdit, QFileDialog, QTabWidget)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QDateTime
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 定义EMGClassifier模型
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

# 定义自定义数据集类
class EMGDataset(Dataset):
    def __init__(self, csv_file, label):
        self.data = np.genfromtxt(csv_file, delimiter=',')
        self.label = label
        self.cycle_length = 200

    def __len__(self):
        return len(self.data) // self.cycle_length

    def __getitem__(self, idx):
        start = idx * self.cycle_length
        end = start + self.cycle_length
        cycle = self.data[start:end].reshape(-1, 1)
        return torch.FloatTensor(cycle), torch.FloatTensor([self.label])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainThread(QThread):
    update_signal = pyqtSignal(float)
    finished_signal = pyqtSignal()
    update_plot_signal = pyqtSignal(list)

    def __init__(self, train_loader):
        super().__init__()
        self.train_loader = train_loader
        self.train_losses = []

    def run(self):
        model = EMGClassifier(input_size=1, hidden_size=128, num_layers=4, output_size=1).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        num_epochs = 200

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)
            self.update_signal.emit(avg_loss)
            self.update_plot_signal.emit(self.train_losses)

        torch.save(model.state_dict(), 'emg_classifier.pth')
        self.finished_signal.emit()

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

class EMGThread(QThread):
    data_updated = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    action_updated = pyqtSignal(str, float, str)

    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self.running = False
        self.buffer = np.zeros(500)
        self.filtered_buffer = np.zeros(500)
        self.timestamps = np.zeros(500)

        # 加载预训练模型
        self.model = EMGClassifier(input_size=1, hidden_size=128, num_layers=4, output_size=1)
        self.model.load_state_dict(torch.load('emg_classifier.pth', map_location=torch.device('cpu')))
        self.model.eval()

        # 定时器用于每秒钟输出四次检测结果
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_data)
        self.timer.start(250)  # 250毫秒触发一次，相当于每秒钟4次

    def run(self):
        self.running = True
        self.start_time = QDateTime.currentDateTime().toMSecsSinceEpoch()
        self.buffer = np.zeros(500)
        self.filtered_buffer = np.zeros(500)
        self.timestamps = np.zeros(500)

        while self.running:
            if self.serial_port.inWaiting() > 0:
                data = self.serial_port.read(2)
                value = int.from_bytes(data, byteorder='big', signed=True)
                current_time = QDateTime.currentDateTime().toMSecsSinceEpoch()
                timestamp = (current_time - self.start_time) / 1000.0

                self.buffer = np.roll(self.buffer, -1)
                self.timestamps = np.roll(self.timestamps, -1)
                self.buffer[-1] = value
                self.timestamps[-1] = timestamp

                # 对整个buffer进行滤波


                self.data_updated.emit(self.buffer, self.buffer, self.timestamps)

    def stop(self):
        self.running = False
        self.timer.stop()

    def process_data(self):
        if len(self.buffer) >=200:
            # 使用的最新200个采样点
            data_for_prediction = self.buffer[-200:].reshape(-1, 1)
            action, confidence = self.predict_action(data_for_prediction)
            timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz")
            self.action_updated.emit(action, confidence, timestamp)

    def predict_action(self, emg_data):
        # 创建数组的副本以确保连续的内存布局
        emg_data_copy = np.array(emg_data, copy=True)
        emg_tensor = torch.FloatTensor(emg_data_copy).unsqueeze(0)
        with torch.no_grad():
            output = self.model(emg_tensor)
        action = "握手" if output.item() < 0.5 else "举手"
        confidence = output.item() if output.item() > 0.5 else 1 - output.item()
        return action, confidence

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG动作识别上位机")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.train_tab = QWidget()
        self.emg_tab = QWidget()
        self.tabs.addTab(self.train_tab, "训练")
        self.tabs.addTab(self.emg_tab, "实时显示")

        self.setup_train_tab()
        self.setup_emg_tab()

        self.dataset1 = None
        self.dataset2 = None
        self.serial_port = None
        self.emg_thread = None

        self.history_text.append(f"请导入训练集")

    def setup_train_tab(self):
        layout = QVBoxLayout(self.train_tab)

        # 训练数据导入
        data_layout = QHBoxLayout()
        self.import_button1 = QPushButton("选择训练集1(握手)文件")
        self.import_button1.clicked.connect(self.load_dataset1)
        self.import_button2 = QPushButton("选择训练集2(举手)文件")
        self.import_button2.clicked.connect(self.load_dataset2)
        self.start_button = QPushButton("开始训练")
        self.start_button.clicked.connect(self.start_training)
        self.start_button.setEnabled(False)
        data_layout.addWidget(self.import_button1)
        data_layout.addWidget(self.import_button2)
        data_layout.addWidget(self.start_button)
        layout.addLayout(data_layout)

        # 添加训练损失曲线显示
        self.train_plot = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.train_plot)

        # 显示训练历史记录
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        layout.addWidget(self.history_text)

    def setup_emg_tab(self):
        layout = QVBoxLayout(self.emg_tab)

        # 串口选择
        serial_layout = QHBoxLayout()
        self.serial_label = QLabel("选择串口：")
        self.serial_combobox = QComboBox()
        self.serial_combobox.addItems(["COM3", "COM4", "COM5", "COM6", "COM7"])
        self.serial_button = QPushButton("连接")
        self.serial_button.clicked.connect(self.connect_serial)
        serial_layout.addWidget(self.serial_label)
        serial_layout.addWidget(self.serial_combobox)
        serial_layout.addWidget(self.serial_button)
        layout.addLayout(serial_layout)

        # 实时数据显示
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.curve = self.plot_widget.plot(pen='b', name='EMG信号')
        layout.addWidget(self.plot_widget)

        # 实时显示预测动作和置信度
        self.action_label = QLabel("动作：")
        self.confidence_label = QLabel("置信度：")
        self.timestamp_label = QLabel("时间戳：")
        layout.addWidget(self.action_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.timestamp_label)

    def load_dataset1(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择训练集1(握手)文件", "", "CSV Files (*.csv)")
        if file_path:
            self.dataset1 = EMGDataset(file_path, label=0)
            self.history_text.append(f"已加载训练集1: {file_path}")
            if self.dataset2:
                self.start_button.setEnabled(True)

    def load_dataset2(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择训练集2(举手)文件", "", "CSV Files (*.csv)")
        if file_path:
            self.dataset2 = EMGDataset(file_path, label=1)
            self.history_text.append(f"已加载训练集2: {file_path}")
            if self.dataset1:
                self.start_button.setEnabled(True)

    def start_training(self):
        if self.dataset1 is None or self.dataset2 is None:
            self.history_text.append("请先加载训练集")
            return

        combined_dataset = torch.utils.data.ConcatDataset([self.dataset1, self.dataset2])
        train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
        self.train_thread = TrainThread(train_loader)
        self.train_thread.update_signal.connect(self.update_loss)
        self.train_thread.finished_signal.connect(self.training_finished)
        self.train_thread.update_plot_signal.connect(self.update_plot)
        self.train_thread.start()
        self.history_text.append("开始训练...")

    def update_loss(self, loss):
        self.history_text.append(f"训练损失: {loss:.4f}")

    def training_finished(self):
        self.history_text.append("训练完成!")

    def update_plot(self, losses):
        self.train_plot.axes.clear()
        self.train_plot.axes.plot(losses)
        self.train_plot.draw()

    def connect_serial(self):
        port_name = self.serial_combobox.currentText()
        try:
            self.serial_port = serial.Serial(port_name, 9600, timeout=1)
            self.history_text.append(f"已连接到串口 {port_name}")
            self.serial_button.setEnabled(False)

            # 启动EMG线程
            self.emg_thread = EMGThread(self.serial_port)
            self.emg_thread.data_updated.connect(self.update_plot_data)
            self.emg_thread.action_updated.connect(self.update_action)
            self.emg_thread.start()
        except Exception as e:
            self.history_text.append(f"连接串口失败: {str(e)}")

    def update_plot_data(self, raw_data, filtered_data, timestamps):
        self.curve.setData(timestamps, raw_data)

    def update_action(self, action, confidence, timestamp):
        self.action_label.setText(f"动作: {action}")
        self.confidence_label.setText(f"置信度: {confidence:.2f}")
        self.timestamp_label.setText(f"时间戳: {timestamp}")

    def closeEvent(self, event):
        if self.emg_thread and self.emg_thread.running:
            self.emg_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
