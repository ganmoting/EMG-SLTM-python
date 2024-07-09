#include <WiFi.h>

const char *ssid = "mate40pro"; // WiFi 名
const char *password = "12345678"; // WiFi 密码

const IPAddress serverIP(192, 168, 43, 80); // 欲访问的服务端 IP 地址
uint16_t serverPort = 56050; // 服务端口号

WiFiClient client; // 声明一个 ESP32 客户端对象，用于与服务器进行连接

void setup() {
    Serial.begin(115200);
    Serial.println();

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false); // 关闭 STA 模式下 WiFi 休眠，提高响应速度
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("Connected");
    Serial.print("IP Address:");
    Serial.println(WiFi.localIP());
}

void loop() {
    if (client.connect(serverIP, serverPort)) {
        Serial.println("Connected to server");
        for (int i = 1; i <= 10000; ++i) {
            client.println(i);  // 使用println确保每个数字都在新的一行
            Serial.printf("Sent: %d\n", i);
            delay(10);  // 可以根据需要调整延迟
        }
        client.stop();
    }
    delay(1000);  // 等待一段时间后再次尝试连接
}