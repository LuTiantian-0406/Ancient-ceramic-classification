import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('code\\img2.png')
cv2.imshow("img", img)
cv2.waitKey(0)    
cv2.destroyAllWindows() 
