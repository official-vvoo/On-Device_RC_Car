import os
import cv2
from time import sleep

CONST_EXT_IMAGE = ".jpg"

CONST_PATH_DATA = os.path.dirname(os.path.abspath(__file__))
CONST_PATH_IMAGE = os.path.join(CONST_PATH_DATA, "data", "images")

if os.path.exists(CONST_PATH_IMAGE):
    pass
else:
    os.mkdir(CONST_PATH_IMAGE)

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    n = 0

    while True:
        ret, frame = cap.read()

        rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        image_name = str(n)+CONST_EXT_IMAGE

        try:
            cv2.imshow("video", rotated_frame)
        except:
            pass
        
        cv2.imwrite(os.path.join(CONST_PATH_IMAGE, image_name), rotated_frame)
        
        sleep(1)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        
        n+=1
    
    cv2.destroyAllWindows()
