import cv2
import mediapipe as mp
import subprocess

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


def switch_to_desktop(monitor: str, desktop_number: int):
    # bspc десктопы начинаются с 1
    try:
        subprocess.run(["bspc", "desktop", "-f", f"^{desktop_number}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при переключении: {e}")

# Пример вызова


with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        finger_count = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = hand_handedness.classification[0].label
                lm = hand_landmarks.landmark

                # Идентификаторы для указательного, среднего, безымянного, мизинца
                finger_tips = [8, 12, 16, 20]
                finger_dips = [6, 10, 14, 18]

                is_fist = True

                for tip, dip in zip(finger_tips, finger_dips):
                    if lm[tip].y < lm[dip].y:
                        finger_count += 1
                        is_fist = False  # Если хотя бы один палец поднят — не кулак

                # Отображение результата
                if is_fist:
                    cv2.putText(frame, "Fist", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    switch_to_desktop("", 5)  # или просто switch_to_desktop("", 3)
                else:
                    cv2.putText(frame, f"Fingers: {finger_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    switch_to_desktop("", finger_count)  # или просто switch_to_desktop("", 3)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Finger Counter with Fist Detection', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
