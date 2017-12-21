import numpy as np
import cv2

cap = cv2.VideoCapture(0)
i = 6
# while(True):
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frameq
    cv2.imshow('color',frame)
    # cv2.imshow('gray',gray)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    k = cv2.waitKey(1)

    if k & 0xFF == ord('q') or k %256 == 27:
        # ESC pressed
        print("Escape/q hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(i)
        cv2.imshow(img_name, frame)
        while (True):

            k2 = cv2.waitKey(1)
            if k2 & 0xFF == ord('s'):
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                i += 1
                cv2.destroyWindow(img_name)
                break

            elif k2 & 0xFF == ord('q') or k2 %256 == 27:
                # ESC pressed
                cv2.destroyWindow(img_name)
                print("{} not saved!".format(img_name))
                break


    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
