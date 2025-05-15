import cv2
import mediapipe as mp

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmarks based on MediaPipe Hand model
finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
thumb_tip = 4
thumb_ip = 3  # Intermediate joint of thumb for better accuracy

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    total_fingers = 0  # Counter for all hands

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Store landmark positions
            landmark_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                height, width, _ = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                landmark_list.append((cx, cy))

            # Ensure landmarks are detected
            if landmark_list:
                fingers = []

                # Thumb logic (compares tip with IP joint for better accuracy)
                if landmark_list[thumb_tip][0] > landmark_list[thumb_ip][0]:  # Thumb is extended
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Four fingers logic (Index, Middle, Ring, Pinky)
                for i in range(1, 5):
                    if landmark_list[finger_tips[i]][1] < landmark_list[finger_tips[i] - 2][1]:  
                        fingers.append(1)  # Finger is extended
                    else:
                        fingers.append(0)  # Finger is closed

                # Total fingers counted
                total_fingers += sum(fingers)

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display total fingers count
    cv2.putText(img, f'Fingers: {total_fingers}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("Finger Counter", img)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

