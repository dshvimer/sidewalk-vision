import numpy as np
import cv2

cap = cv2.VideoCapture('stroll.mp4')
scale_percent = 33

while(cap.isOpened()):
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # img = frame

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    Z = resized.reshape((-1,3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 3, 1.0)
    K = 2
    ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((resized.shape))

    cv2.imshow('frame',res2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
