

# EMG Action Recognition Upper Computer

This project is a Python-based upper computer program for EMG (Electromyography) action recognition. It can real-time collect EMG signals, perform filtering, and utilize a pretrained deep learning model for action recognition.

![Pattern recognition training upper computer](test/7ac0a7f9a7262d5185a2f7b9bb3fd76.png)

![Pattern recognition training upper computer](test/e21d7785afc1c6f7ab478c9102e746d.png)

## Key Features

- Real-time EMG signal collection and display
- Signal filtering
- Action recognition based on LSTM
- Intuitive graphical user interface
- Real-time display of action recognition results and historical records

## Technology Stack

- Python 3.8
- PyQt5: For building the graphical user interface
- PyQtGraph: For real-time data visualization
- PySerial: For serial communication
- NumPy: For numerical computing
- SciPy: For signal processing
- PyTorch: For deep learning models

## Installation

1. `pip install PyQt5`
2. `pip install PySerial`
3. `pip install PyQtGraph`
4. `pip install NumPy`
5. `pip install SciPy`
6. `pip install PyTorch`

## Usage

1. Run the main program:
2. Select the correct serial port on the interface and click "Open Serial Port"
3. Observe real-time EMG signals and filtered signals
4. View action recognition results and historical records

## Project Structure

- `new1_shangweiji.py`: Main program entry point
- `train3_best.py`: LSTM model training
- `signal_processing.py`: Signal processing functions
- `train_shangweiji.py`: Graphical user interface definition
- `emg_classifier.pth`: Pretrained model weights
  ![模式识别训练上位机](test/0379543d173e56d2f6b33c09e46c66a.png)
  ![模式识别训练上位机](test/815eebdcc6af66670ad318891cc8d24.png)
  ![模式识别训练上位机](test/8604311dde7f3d41c2b2f2e82b8f816.png)

## Contribution

Feel free to submit issues and pull requests.


