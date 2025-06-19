import cv2
import mediapipe as mp
import time

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Load video
video = cv2.VideoCapture('your_video.mp4')  # Change to your video file
if not video.isOpened():
    print("❌ Cannot open video file.")
    exit()

# Start webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Cannot access webcam.")
    exit()

# Setup windows
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Hand Gesture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", 640, 360)
cv2.resizeWindow("Hand Gesture", 320, 240)
cv2.moveWindow("Video", 100, 100)
cv2.moveWindow("Hand Gesture", 800, 100)

# State variables
playing = True
muted = False
last_gesture = None
last_time = time.time()
gesture_cooldown = 1.5  # seconds

def get_fingers_up(hand):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    # Thumb: left/right
    fingers.append(1 if hand.landmark[4].x < hand.landmark[3].x else 0)
    # Other fingers: up/down
    for tip in tip_ids[1:]:
        fingers.append(1 if hand.landmark[tip].y < hand.landmark[tip - 2].y else 0)
    return fingers

while True:
    # Read webcam
    ret_cam, cam_frame = cam.read()
    if not ret_cam:
        break
    cam_frame = cv2.flip(cam_frame, 1)
    cam_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(cam_rgb)

    # Detect gesture
    gesture = None
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(cam_frame, handLms, mp_hands.HAND_CONNECTIONS)
            fingers = get_fingers_up(handLms)

            if fingers == [0, 1, 1, 1, 1]:       gesture = "play"
            elif fingers == [0, 0, 0, 0, 0]:     gesture = "pause"
            elif fingers == [0, 1, 0, 0, 0]:     gesture = "forward"
            elif fingers == [1, 0, 0, 0, 0]:     gesture = "rewind"
            elif fingers == [0, 1, 1, 0, 0]:     gesture = "mute"
            elif fingers == [0, 1, 0, 0, 1]:     gesture = "restart"

    # Trigger gesture only if cooldown passed or new
    current_time = time.time()
    if gesture and (gesture != last_gesture or current_time - last_time > gesture_cooldown):
        fps = video.get(cv2.CAP_PROP_FPS)
        pos = video.get(cv2.CAP_PROP_POS_FRAMES)

        if gesture == "play":
            playing = True
            print("▶️ Play")
        elif gesture == "pause":
            playing = False
            print("⏸ Pause")
        elif gesture == "forward":
            video.set(cv2.CAP_PROP_POS_FRAMES, pos + int(fps * 2))  # 2 seconds forward
            print("⏩ Forward 2s")
        elif gesture == "rewind":
            video.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - int(fps * 2)))  # 2 seconds back
            print("⏪ Rewind 2s")
        elif gesture == "mute":
            muted = not muted
            print("🔇 Muted" if muted else "🔊 Unmuted")
        elif gesture == "restart":
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video.read()  # drop 1 frame to prevent freeze
            playing = True
            print("⏮ Restarted")

        last_gesture = gesture
        last_time = current_time

    # Show webcam feed
    cam_frame = cv2.resize(cam_frame, (320, 240))
    cv2.imshow("Hand Gesture", cam_frame)

    # Show video
    if playing:
        ret_vid, frame = video.read()
        if not ret_vid:
            break
        if muted:
            frame[:, :, 1] = 0  # visually indicate mute
        frame = cv2.resize(frame, (640, 360))
        cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cam.release()
video.release()
cv2.destroyAllWindows()
