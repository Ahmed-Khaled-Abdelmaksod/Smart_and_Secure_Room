import cv2

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open camera")
    else:
        ret , frame = cap.read()
        if ret:
            cv2.imshow("Preview", frame)
            cv2.waitKey(0)  # Wait until key press
            cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()