import cv2
import mediapipe as mp
import math
import subprocess

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand_tracker = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Константы для настройки
THUMB_TIP = 4
INDEX_TIP = 8
VOLUME_HAND = "Left"
BRIGHTNESS_HAND = "Right"
DISTANCE_RANGE = (0.05, 0.3)
LERP_SPEED = 0.3

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def normalize_value(distance, min_val=0, max_val=0.3):
    clamped = max(min_val, min(distance, max_val))
    return int(((clamped - min_val) / (max_val - min_val)) * 100)

def set_system_value(command, value):
    try:
        subprocess.run(command + [f'{value}%'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка выполнения команды: {e}")

def linear_interpolate(current, target, speed):
    return current + (target - current) * speed

def draw_control_line(frame, start, end):
    h, w, _ = frame.shape
    cv2.line(frame,
             (int(start.x * w), int(start.y * h)),
             (int(end.x * w), int(end.y * h)),
             (255, 0, 0), 3)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: камера недоступна")
        return

    # Текущие значения с плавным изменением
    volume = brightness = 50
    target_volume = target_brightness = 100

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Подготовка кадра
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обработка рук
        results = hand_tracker.process(rgb_frame)
        if results.multi_hand_landmarks:
            for landmarks, handedness in zip(results.multi_hand_landmarks,
                                            results.multi_handedness):
                hand_type = handedness.classification[0].label
                thumb = landmarks.landmark[THUMB_TIP]
                index = landmarks.landmark[INDEX_TIP]

                # Визуализация
                draw_control_line(frame, thumb, index)
                mp_drawing.draw_landmarks(
                    frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Обновление целей управления
                distance = calculate_distance(thumb, index)
                value = normalize_value(distance, *DISTANCE_RANGE)

                if hand_type == VOLUME_HAND:
                    target_volume = value
                elif hand_type == BRIGHTNESS_HAND:
                    target_brightness = value

        # Плавное изменение значений
        volume = linear_interpolate(volume, target_volume, LERP_SPEED)
        brightness = linear_interpolate(brightness, target_brightness, LERP_SPEED)

        # Применение изменений в системе
        set_system_value(['amixer', 'set', 'Master'], volume)
        set_system_value(['brightnessctl', 'set'], brightness)

        # Отображение информации
        cv2.putText(frame, f'Volume: {int(volume)}%',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Brightness: {int(brightness)}%',
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
