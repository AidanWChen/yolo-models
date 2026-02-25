from ultralytics import YOLO
import cv2 as cv

bottlem = YOLO("bottle.pt")
hammerm = YOLO("hammer.pt")
pickm = YOLO("pick.pt")

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()

    combined_results = [ bottlem(frame), hammerm(frame),pickm(frame)]
    
    annotated_frame = combined_results[0][0].plot()
    annotated_frame = combined_results[1][0].plot(img=annotated_frame)
    annotated_frame = combined_results[2][0].plot(img=annotated_frame)

    cv.imshow("YOLO Object Detection", annotated_frame) # Display the annotated frame

    if cv.waitKey(1) == ord('q'): # Press 'q' to quit
        break

capture.release() # Release the camera
cv.destroyAllWindows() # Close all OpenCV windows    
