import socket
import json
import time
import datetime
import cv2
import numpy as np

# Setup untuk YOLO
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Ganti dengan path yang sesuai
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Inisialisasi socket
SUBSCRIBER_IP = "ganti" 
PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SUBSCRIBER_IP, PORT))
print("Connected to subscriber")

msg_id = 1
cap = cv2.VideoCapture(0)  # Ambil input dari webcam (bisa ganti dengan file video)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Deteksi objek dengan YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)

    # Proses deteksi objek (opsional, bisa ditambah untuk mendeteksi objek spesifik)
    # Misalnya hanya untuk mendeteksi orang
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Ambil deteksi dengan confidence tinggi
                # Kirim frame yang terdeteksi ke subscriber
                ret, buffer = cv2.imencode('.jpg', frame)
                message = {
                    "nama": "Publisher",
                    "user_id": "PUB001",
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "day": datetime.datetime.now().strftime("%A"),
                    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "message_id": msg_id,
                    "frame": buffer.tobytes()  # Kirim gambar dalam format bytes
                }
                sock.sendall((json.dumps(message) + "\n").encode())
                msg_id += 1
                time.sleep(2)  # Delay agar tidak mengirim terlalu cepat

    cv2.imshow('YOLO Detection', frame)
    
    # Keluar jika tekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
