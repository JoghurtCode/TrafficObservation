import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture('cars.mp4')
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    car_count = 0
    min_contour_width = 20
    min_contour_height = 20
    min_contour_area = 300
    max_contour_area = 50000

    counting_line_position = 550
    offset = 10

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    counted_contours = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = backSub.apply(frame)
        _, thresh_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        morph_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel, iterations=3)
        morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if min_contour_area < cv2.contourArea(contour) < max_contour_area and 0.2 < aspect_ratio < 4.0:
                if w >= min_contour_width and h >= min_contour_height:
                    color = (0, 255, 0)
                    label = "Counted"
                    car_count += 1
                    
                    # if (counting_line_position - offset) < y + h < (counting_line_position + offset):
                    #     if contour not in counted_contours:
                    #         car_count += 1
                    #         counted_contours.append(contour)
                else:
                    color = (0, 0, 255)
                    label = "Too Small"
            else:
                color = (0, 0, 255)
                label = "Not Car"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        cv2.line(frame, (0, counting_line_position), (frame.shape[1], counting_line_position), (255, 0, 0), 2)
        cv2.putText(frame, f'Car Count: {car_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Display the different masks
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Foreground Mask', fg_mask)
        cv2.imshow('Thresholded Mask', thresh_mask)
        cv2.imshow('Morphologically Processed Mask', morph_mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
