from extract import extractIcons
import numpy
import time
import cv2
import mss

monitor = {'top': 1061, 'left': 2216, 'width': 327, 'height': 322}

count = 0
with mss.mss() as sct:
    time.sleep(15)
    while True:
        img = numpy.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        a, e = extractIcons(img)

        icons = a + e
        if len(icons) != 0: 
            for i, icon in enumerate(icons):
                try:
                    cv2.imwrite(f'data/unlabeled/{count}.jpg', icon)
                except Exception as e:
                    print('Error', e)
                count += 1
        time.sleep(5)
