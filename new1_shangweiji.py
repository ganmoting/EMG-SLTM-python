import sys
import numpy as np
import torch
import torch.nn as nn
import serial
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, \
    QWidget, QTextEdit
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QDateTime
from scipy.signal import butter, filtfilt


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
        self.buffer = np.zeros(10000)
        self.filtered_buffer = np.zeros(10000)
        self.timestamps = np.zeros(10000)

        # 加载预训练模型
        self.model = EMGClassifier(input_size=1, hidden_size=64, num_layers=3, output_size=1)
        self.model.load_state_dict(torch.load('emg_classifier.pth', map_location=torch.device('cpu')))
        self.model.eval()

        # 定时器用于每秒钟输出四次检测结果
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_data)
        self.timer.start(250)  # 250毫秒触发一次，相当于每秒钟4次

    def run(self):
        self.running = True
        self.start_time = QDateTime.currentDateTime().toMSecsSinceEpoch()
        self.buffer = np.zeros(10000)
        self.filtered_buffer = np.zeros(10000)
        self.timestamps = np.zeros(10000)

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
                self.filtered_buffer = butter_lowpass_filter(self.buffer, cutoff=50, fs=1000, order=6)

                self.data_updated.emit(self.buffer, self.filtered_buffer, self.timestamps)

    def stop(self):
        self.running = False
        self.timer.stop()

    def process_data(self):
        if len(self.filtered_buffer) == 10000:
            # 确保我们使用的是最新的1000个采样点
            data_for_prediction = self.filtered_buffer[-1000:].reshape(-1, 1)
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
        self.setGeometry(100, 100, 1000, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 串口控制
        port_layout = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.addItems(['COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6'])
        self.connect_button = QPushButton("打开串口")
        self.connect_button.clicked.connect(self.toggle_connection)
        port_layout.addWidget(self.port_combo)
        port_layout.addWidget(self.connect_button)
        layout.addLayout(port_layout)

        # 原始波形图
        self.raw_plot_widget = pg.PlotWidget(title="原始EMG信号")
        self.raw_plot_curve = self.raw_plot_widget.plot(pen='y')
        layout.addWidget(self.raw_plot_widget)

        # 滤波后波形图
        self.filtered_plot_widget = pg.PlotWidget(title="滤波后EMG信号")
        self.filtered_plot_curve = self.filtered_plot_widget.plot(pen='g')
        layout.addWidget(self.filtered_plot_widget)

        # 放大和缩小按钮
        zoom_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("放大")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button = QPushButton("缩小")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.clear_button = QPushButton("清除")
        self.clear_button.clicked.connect(self.clear_plot)
        zoom_layout.addWidget(self.zoom_in_button)
        zoom_layout.addWidget(self.zoom_out_button)
        zoom_layout.addWidget(self.clear_button)
        layout.addLayout(zoom_layout)

        # 动作识别结果
        self.action_label = QLabel("识别结果: ")
        layout.addWidget(self.action_label)

        # 动作识别历史记录
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        layout.addWidget(self.history_text)

        self.serial_port = None
        self.emg_thread = None

    def toggle_connection(self):
        if self.serial_port is None:
            port = self.port_combo.currentText()
            try:
                self.serial_port = serial.Serial(port, 115200)
                self.connect_button.setText("关闭串口")
                self.emg_thread = EMGThread(self.serial_port)
                self.emg_thread.data_updated.connect(self.update_plot)
                self.emg_thread.action_updated.connect(self.update_action)
                self.emg_thread.start()
            except serial.SerialException:
                self.action_label.setText("无法打开串口")
        else:
            self.emg_thread.stop()
            self.emg_thread.wait()
            self.serial_port.close()
            self.serial_port = None
            self.connect_button.setText("打开串口")

    def update_plot(self, raw_data, filtered_data, timestamps):
        self.raw_plot_curve.setData(timestamps, raw_data)
        self.filtered_plot_curve.setData(timestamps, filtered_data)

    def update_action(self, action, confidence, timestamp):
        self.action_label.setText(f"识别结果: {action} (置信度: {confidence:.2f})")
        self.history_text.append(f"{timestamp} - 识别结果: {action} (置信度: {confidence:.2f})")

    def zoom_in(self):
        self.raw_plot_widget.getViewBox().scaleBy((0.5, 1))
        self.filtered_plot_widget.getViewBox().scaleBy((0.5, 1))

    def zoom_out(self):
        self.raw_plot_widget.getViewBox().scaleBy((2, 1))
        self.filtered_plot_widget.getViewBox().scaleBy((2, 1))

    def clear_plot(self):
        self.raw_plot_curve.clear()
        self.filtered_plot_curve.clear()
        self.emg_thread.buffer = np.zeros(10000)
        self.emg_thread.filtered_buffer = np.zeros(10000)
        self.emg_thread.timestamps = np.zeros(10000)
        self.emg_thread.start_time = QDateTime.currentDateTime().toMSecsSinceEpoch()
        self.update_plot(self.emg_thread.buffer, self.emg_thread.filtered_buffer, self.emg_thread.timestamps)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())