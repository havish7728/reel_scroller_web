import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_y = None
gesture_cooldown = 1.5  # seconds between actions
last_gesture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    frame = cv2.flip(frame, 1)  # mirror image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        index_tip = hand_landmarks.landmark[8]
        finger_y = index_tip.y

        if prev_y is not None:
            diff = prev_y - finger_y
            current_time = time.time()
            if current_time - last_gesture_time > gesture_cooldown:
                if diff > 0.1:
                    print("Gesture detected: Swipe UP → pressing Up arrow")
                    pyautogui.press('up')
                    last_gesture_time = current_time
                    cv2.putText(frame, 'Up Arrow Pressed', (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif diff < -0.1:
                    print("Gesture detected: Swipe DOWN → pressing Down arrow")
                    pyautogui.press('down')
                    last_gesture_time = current_time
                    cv2.putText(frame, 'Down Arrow Pressed', (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        prev_y = finger_y
    else:
        prev_y = None  # reset if no hand detected

    cv2.imshow("Reel Controller - Swipe Up/Down", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
