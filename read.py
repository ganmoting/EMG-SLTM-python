import scipy.io as sio
import matplotlib.pyplot as plt

# Load the MAT file
mat_file_path = 'test/matlab.mat'
mat_data = sio.loadmat(mat_file_path)

# Extract the EMG data
emg_data = mat_data['EMG']

# Plot the waveform
plt.figure(figsize=(50, 10))
plt.plot(emg_data)
plt.title('EMG Data Waveform')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
