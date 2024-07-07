import serial
import time
import struct
import csv
import itertools


def read_csv_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # 假设CSV文件只有一列数据
        return [int(float(row[0])) for row in reader]


def main():
    # 打开COM5串口
    ser = serial.Serial('COM5', 115200, timeout=1)

    # 读取CSV文件中的数据
    data = read_csv_data('E:\demo\data\strain\stest1.csv')

    try:
        # 使用itertools.cycle来循环数据
        for value in itertools.cycle(data):
            # 将值转换为2字节的有符号整数
            data_bytes = struct.pack('>h', value)
            ser.write(data_bytes)
            time.sleep(0.0000001)  # 1000Hz的采样率，可以根据需要调整
    except KeyboardInterrupt:
        print("程序已停止")
    finally:
        ser.close()


if __name__ == "__main__":
    main()