# EMG动作识别上位机

这个项目是一个基于Python的EMG(肌电图)动作识别上位机程序。它能够实时采集EMG信号,进行滤波处理,并利用预训练的深度学习模型进行动作识别。

## 功能特点

- 实时EMG信号采集和显示
- 信号滤波处理
- 基于LSTM的动作识别
- 直观的图形用户界面
- 动作识别结果实时显示和历史记录

## 技术栈

- Python 3.8
- PyQt5: 用于构建图形用户界面
- PyQtGraph: 用于实时数据可视化
- PySerial: 用于串口通信
- NumPy: 用于数值计算
- SciPy: 用于信号处理
- PyTorch: 用于深度学习模型

## 安装

1. `pip install PyQt5`
2. `pip install PySerial`
3. `pip install PyQtGraph`
4. `pip install NumPy`
5. `pip install SciPy`
6. `pip install PyTorch`
  
## 使用方法

1. 运行主程序:
2. 在界面上选择正确的串口并点击"打开串口"
3. 观察实时EMG信号和滤波后的信号
4. 查看动作识别结果和历史记录

## 项目结构

- `new1_shangweiji.py`: 主程序入口
- `train3_best.py`: SLTM训练模型
- `signal_processing.py`: 信号处理函数
- `train_shangweiji.py`: 图形用户界面定义
- `emg_classifier.pth`: 预训练模型权重
  
  ## 文件结构
- `data`: 数据集文件夹，strain为训练集，val为测试集，`model_bo_com1.py`为从串口读入EMG信号
- `xx2xx.py`：命名规则为xx2yy，作用为将文件从xx格式转为yy格式
- `train_shanghweiji.py`：作用为创建一个可视化界面，使代码更加易懂
  
## 贡献

欢迎提交问题和合并请求。telephone：+86 13389947952
