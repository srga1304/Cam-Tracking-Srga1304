import cv2
import numpy as np

# Настройки
KNOWN_WIDTH = 125    # Ширина калибровочного объекта (мм) - банковская карта
KNOWN_HEIGHT = 78  # Высота калибровочного объекта (мм)
CALIBRATION_KEY = 'c' # Клавиша для калибровки
QUIT_KEY = 'q'        # Клавиша для выхода
DEBUG_MODE = False    # Режим отладки (показывает промежуточные этапы обработки)

# Глобальные переменные
pixels_per_metric = None
calibrated = False

def detect_objects(image):
    """Обнаружение объектов на изображении с улучшенной обработкой для темных объектов"""
    # Переход в HSV и выделение канала насыщенности
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:,:,1]

    # Усиление контраста с помощью CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_s = clahe.apply(s_channel)

    # Адаптивная бинаризация для темных объектов
    binary = cv2.adaptiveThreshold(
        enhanced_s, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,  # Размер блока
        5    # Константа C
    )

    # Морфологическая обработка
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # Поиск контуров
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация мелких шумов
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 500]

    # Отладочная визуализация
    if DEBUG_MODE:
        debug_img = cv2.cvtColor(enhanced_s, cv2.COLOR_GRAY2BGR)
        debug_img = cv2.drawContours(debug_img, filtered_contours, -1, (0,255,0), 2)
        cv2.imshow("Debug", debug_img)

    return filtered_contours

def find_main_contour(contours):
    """Находит основной контур среди обнаруженных"""
    if not contours:
        return None

    # Сортируем контуры по площади (от большего к меньшему)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0]

def calibrate(image, contour):
    """Выполняет калибровку по известному объекту"""
    global pixels_per_metric, calibrated
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Вычисляем среднюю длину сторон
    (tl, tr, br, bl) = box
    width_pixels = np.linalg.norm(tr - tl)
    height_pixels = np.linalg.norm(bl - tl)
    avg_pixels = (width_pixels + height_pixels) / 2.0

    # Рассчитываем коэффициент преобразования (пиксели на мм)
    pixels_per_metric = avg_pixels / ((KNOWN_WIDTH + KNOWN_HEIGHT) / 2)
    calibrated = True

    # Рисуем контур калибровочного объекта
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    cv2.putText(image, "Calibrated!", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image

def measure_object(image, contour):
    """Измеряет объект и отображает размеры"""
    if not calibrated:
        return image

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Рисуем контур объекта
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    # Вычисляем размеры
    (tl, tr, br, bl) = box
    width_pixels = min(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    height_pixels = min(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))

    # Конвертируем пиксели в миллиметры
    width_mm = width_pixels / pixels_per_metric
    height_mm = height_pixels / pixels_per_metric

    # Отображаем размеры
    cv2.putText(image, "W: {:.1f}mm".format(width_mm),
                (int(tl[0]), int(tl[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(image, "H: {:.1f}mm".format(height_mm),
                (int(tr[0]), int(tr[1] + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return image

def main():
    global calibrated, DEBUG_MODE

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    print("Инструкция:")
    print("1. Поместите калибровочный объект (банковскую карту) перед камерой")
    print("2. Нажмите 'c' для калибровки")
    print("3. Поместите измеряемый объект перед камерой")
    print("4. Нажмите 'd' для включения/выключения режима отладки")
    print("5. Нажмите 'q' для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр с камеры")
            break

        # Масштабирование для более быстрой обработки
        scale = 0.7
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # Обнаружение объектов
        contours = detect_objects(frame)
        contour = find_main_contour(contours)

        if contour is not None:
            if calibrated:
                frame = measure_object(frame, contour)
            else:
                # Проверяем, что объект прямоугольный
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    frame = calibrate(frame, contour)

        # Отображаем инструкции
        if not calibrated:
            cv2.putText(frame, "Press '{}' to calibrate".format(CALIBRATION_KEY),
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "Calibrated - Ready to measure",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Measurement Tool", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(CALIBRATION_KEY) and contour is not None:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                frame = calibrate(frame, contour)
        elif key == ord('d'):
            DEBUG_MODE = not DEBUG_MODE
            if DEBUG_MODE:
                print("Режим отладки включен")
            else:
                print("Режим отладки выключен")
                cv2.destroyWindow("Debug")
        elif key == ord(QUIT_KEY):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
