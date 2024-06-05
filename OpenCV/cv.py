import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture('cars.mp4')
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    car_count = 0
    min_contour_width = 40
    min_contour_height = 40
    min_contour_area = 500
    max_contour_area = 50000

    counting_line_position = 550
    offset = 10

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = backSub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=3)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if min_contour_area < cv2.contourArea(contour) < max_contour_area and w >= min_contour_width and h >= min_contour_height and 0.2 < aspect_ratio < 4.0:
                color = (0, 255, 0)  # Green for detected and counted cars
                if (y + h) > (counting_line_position - offset) and (y + h) < (counting_line_position + offset):
                    car_count += 1
            else:
                color = (0, 0, 255)  # Red for detected but not counted cars

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.line(frame, (0, counting_line_position), (frame.shape[1], counting_line_position), (255, 0, 0), 2)
        cv2.putText(frame, f'Car Count: {car_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
