import socket
import serial
import time

def main():
    ser = serial.Serial('COM5', 115200, timeout=1)
    server_ip = '0.0.0.0'
    server_port = 56050

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(1)
    print(f'Server listening on {server_ip}:{server_port}')

    while True:
        client_socket, client_address = server_socket.accept()
        print(f'Connection from {client_address}')

        try:
            buffer = ""
            while True:
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    try:
                        value = int(line.strip())
                        ser.write(value.to_bytes(2, byteorder='big'))
                        #print(f'Sent to COM5: {value}')
                    except ValueError as e:
                        print(f'Error: {e}')
        except KeyboardInterrupt:
            print("程序已停止")
            break
        finally:
            client_socket.close()

    server_socket.close()
    ser.close()

if __name__ == "__main__":
    main()