import numpy as np
import cv2
import os
import glob
import math

nname = ''
line = ''
eye = []

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

imglist = glob.glob("*.jpg")
#print str(len(imglist))

for i in range(0,len(imglist)):
	eye = []
	fname = imglist[i]

	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray,10,255,0)

	# find face
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	f = 0
	for (x,y,w,h) in faces:
		f += 1

		#line += fname + ',' + 'face,' + str(x)+','+str(y)+','+str(w)+','+str(h)+'\n'
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		face_gray = gray[y:y+h, x:x+w]
		face_color = img[y:y+h, x:x+w]
		ret,face_bw = cv2.threshold(face_gray,100,255,0)

		cv2.imwrite(fname.replace('april_2','april_face')+'.'+str(f)+'.jpg', face_gray)


		# find contours
		im2, contours, hierarchy = cv2.findContours(face_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		for cnt in contours:
			area = cv2.contourArea(cnt)
			#print area
			if area >100 and area < 1000:

				x1,y1,w1,h1 = cv2.boundingRect(cnt)
				if w1 > h1*2 :
					if y1 < (h)/2 :
						#print y,h,y1
						cv2.rectangle(img,(x+x1,y+y1),(x+x1+w1,y+y1+h1),(0,255,0),2)
						eye.append((x+x1,y+y1))
						eye.append((x+x1+w1,y+y1+h1))

						#print len(eye)
						#print eye

						if len(eye) == 4:

							eye.sort()
							x1 = eye[0][0]
							y1 = eye[0][1]
							x2 = eye[1][0]
							y2 = eye[1][1]
							x3 = eye[2][0]
							y3 = eye[2][1]
							x4 = eye[3][0]
							y4 = eye[3][1]

							#print eye

							e1w = math.hypot(x2 - x1, y2 - y1)
							bw = math.hypot(x3 - x2, y3 - y2)
							e2w = math.hypot(x4 - x3, y4 - y3)
							tw = e1w+bw+e2w

							e1p = float(e1w) / float(tw)
							bp = float(bw) / float(tw)
							e2p = float(e2w) / float(tw)

							line += fname+','
							line += str(e1w)+','
							line += str(bw)+','
							line += str(e2w)+','
							line += str(tw)+','
							line += str(e1p)+','
							line += str(bp)+','
							line += str(e2p)+'\n'

							#print line
						#print x1+w1/2,y1+h1/2

		for z in range(0,len(contours)):
			for zz in range(0,len(contours[z])):
				for zzz in range(0,len(contours[z][zz])):
					contours[z][zz][zzz][0] += x
					contours[z][zz][zzz][1] += y

		cv2.drawContours(img, contours, -1, (255,255,0), 1)

	cv2.imshow('img',img)
	k = cv2.waitKey(0)
	#line += chr(k)+'\n'



line += '\n'
#print line

f = open('april_face_finder.txt', 'w')
f.write(line)
f.close()
cv2.destroyAllWindows()