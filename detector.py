import cv2
import pyttsx3
import threading
import time
from collections import defaultdict
from ultralytics import YOLO

# ============================================
# ভয়েস সেটআপ
# ============================================
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1.0)

def speak(text):
    def _speak():
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=_speak, daemon=True).start()

# ============================================
# মডেল লোড
# ============================================
print("🔄 মডেল লোড হচ্ছে...")
model = YOLO('yolov8n.pt')
print("✅ মডেল রেডি!")

# ============================================
# ফোনের IP এখানে বসাও ⬇️⬇️⬇️
# ============================================
PHONE_IP = "192.168.0.101"   # <-- তোমার IP দিয়ে বদলাও
CAMERA_URL = f"http://{PHONE_IP}:8080/video"

print(f"📷 ক্যামেরা কানেক্ট হচ্ছে: {CAMERA_URL}")
cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("❌ ক্যামেরা খোলা যায়নি!")
    print("   - IP Webcam চালু আছে?")
    print("   - IP ঠিক আছে?")
    print("   - একই WiFi-তে আছো?")
    exit()

print("✅ ক্যামেরা চালু! বন্ধ করতে 'q' চাপো।")

# ============================================
# বাংলা নাম
# ============================================
bangla = {
    'person': 'মানুষ', 'laptop': 'ল্যাপটপ', 'cell phone': 'মোবাইল',
    'chair': 'চেয়ার', 'bottle': 'বোতল', 'cup': 'কাপ',
    'book': 'বই', 'keyboard': 'কীবোর্ড', 'mouse': 'মাউস',
    'tv': 'টিভি', 'car': 'গাড়ি', 'dog': 'কুকুর', 'cat': 'বিড়াল',
    'clock': 'ঘড়ি', 'remote': 'রিমোট', 'backpack': 'ব্যাগ',
    'handbag': 'হ্যান্ডব্যাগ', 'scissors': 'কাঁচি', 'knife': 'ছুরি',
    'spoon': 'চামচ', 'fork': 'কাঁটাচামচ', 'bowl': 'বাটি',
    'banana': 'কলা', 'apple': 'আপেল', 'orange': 'কমলা',
    'dining table': 'টেবিল', 'bed': 'বিছানা', 'sink': 'বেসিন',
    'refrigerator': 'ফ্রিজ', 'microwave': 'মাইক্রোওয়েভ',
    'monitor': 'মনিটর', 'teddy bear': 'টেডি বিয়ার',
}

last_spoken = defaultdict(float)
COOLDOWN = 3.0

# ============================================
# মূল লুপ
# ============================================
frame_count = 0
fps_start = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ ফ্রেম আসছে না, আবার চেষ্টা...")
        time.sleep(0.5)
        continue

    # ছোট করো → ফাস্ট হবে
    frame = cv2.resize(frame, (640, 480))

    # ডিটেকশন
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            name = model.names[cls_id]

            # সবুজ বক্স
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # লেবেল
            label = f"{name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # ভয়েস (cooldown সহ)
            now = time.time()
            if now - last_spoken[name] > COOLDOWN:
                speak(bangla.get(name, name))
                last_spoken[name] = now

    # FPS
    frame_count += 1
    if time.time() - fps_start >= 1.0:
        fps = frame_count
        frame_count = 0
        fps_start = time.time()

    cv2.putText(frame, f"FPS: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Live Object Detection (Press Q to exit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ বন্ধ হয়েছে।")