import cv2
import glob
import os

inputFolder = 'kopi/robusta'
os.mkdir('kopi/64x64/robusta')
i = 0
for img in glob.glob(inputFolder + "/*.jpg"):
    image = cv2.imread(img)
    imgResized = cv2.resize(image, (64, 64))
    cv2.imwrite("kopi/64x64/robusta/robusta%04i.jpg" %i, imgResized)
    i +=1
    cv2.imshow('image', imgResized)
    cv2.waitKey(30)
cv2.destroyAllWindows()

