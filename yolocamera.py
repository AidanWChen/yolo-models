from ultralytics import YOLO
import cv2 as cv

model = YOLO("badbottleandmalletmodel.pt")

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()

    results = model(frame)

    annotated_frame = results[0].plot()
    cv.imshow("YOLO Object Detection", annotated_frame) # Display the annotated frame

    if cv.waitKey(1) == ord('q'): # Press 'q' to quit
        break

capture.release() # Release the camera
cv.destroyAllWindows() # Close all OpenCV windows    