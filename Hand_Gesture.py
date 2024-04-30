# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
from time import sleep
from math import floor

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

start_x, start_y = None, None
is_fist_closed = False
gesture_detected = False

while True:
    success, frame = camera.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    results = hands_detector.process(frame_rgb)

    frame = cv2.flip(frame, 1)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            is_fist_closed = abs(middle_tip.y-wrist.y)

            x, y = middle_tip.x, middle_tip.y

            if start_x is None or start_y is None:
                start_x, start_y = x, y

            dx, dy = x - start_x, y - start_y
            distance_moved = abs(dx) + abs(dy)

            palm_to_camera = wrist.y < index_mcp.y and abs(wrist.x - index_mcp.x) < 0.1

            if distance_moved > 0.1:
                if is_fist_closed < 0.2:
                    print("Fist Closed - Click")
                    continue
                if palm_to_camera:
                    if abs(dx) > abs(dy):
                        direction = "Right" if dx > 0 else "Left"
                    else:
                        direction = "Down" if dy > 0 else "Up"
                    print(f"Swipe {direction} with Palm to Camera")
                else:
                    orientation = "Up" if wrist.y > middle_tip.y else "Down"
                    direction = "Right" if abs(dx) > abs(dy) and dx > 0 else "Left"
                    print(f"Swipe {direction} with Palm Facing {orientation}")

                start_x, start_y = None, None

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()