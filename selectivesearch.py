import cv2

def ss(img):
    # Multi-Threading
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    img