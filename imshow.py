import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('h000.jpg',1)

rows,cols,chans = img.shape
for i in range(1,rows):
	for j in range(1,cols):
		if img[i,j,2] > 100 :
			img[i,j] = 255
		else:
			img[i,j] = 0
	

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print 'test'