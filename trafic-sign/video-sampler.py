import numpy as np
import cv2

cap = cv2.VideoCapture('sign_1.mp4')
# cap = cv2.VideoCapture(0)
itr = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    itr += 1
    cv2.waitKey(100)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # median_img = cv2.medianBlur(gray, 17)
    median_img = cv2.GaussianBlur(gray, (5, 5), 5)
    out_img = cv2.GaussianBlur(frame, (5, 5), 5)

    # cv2.imshow('median', median_img)
    # cv2.imshow('gaussian', gaussian_img)

    circles = cv2.HoughCircles(median_img, cv2.HOUGH_GRADIENT,
                               1, 100, param1=240, param2=80)
    # print(circles)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    if not circles is None:
        circles = np.uint16(np.around(circles))
        # print(circles)

        # Extracting (192, 192) images with circles
        for i in range(len(circles[:, :, 2][0])):
            x, y, r = circles[:, :, :][0][i]
            print(x, y, r)
            r += 20

            if y > r and x > r:
                square = frame[y-r:y+r, x-r:x+r]
                # square = cv2.resize(square, (299, 299))
                # print(square.shape, type(square))
                # cv2.imwrite('sign_classifier/images/6_itr_'+str(itr) +
                #             '_cir_'+str(i)+'.jpeg', square)

        # Drawing circles in video
        for i in circles[0, :]:
            cv2.circle(out_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(out_img, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('median', out_img)

cap.release()
cv2.destroyAllWindows()
