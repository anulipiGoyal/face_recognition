import face_recognition
import numpy as np
import csv
import cv2
import os
from datetime import datetime 

####Data taken from local device
path = './BasicImages'
images = []
classNames = []
myImgList = os.listdir('./BasicImages')
print("===Welcome to face Detection Project===")
print(myImgList)

for cl in myImgList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncode(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def attendance(name):
    with open('Attendance.xlsx','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateStr = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateStr}')


knownEncodeList = findEncode(images)
#print(len(knownEncode))
print('Encoding Complete')


###Catured data from webcam
capture_video = cv2.VideoCapture(0)

while True:
    success, img = capture_video.read()
    imgSmall = cv2.resize(img, (0,0),None, 0.25,0.25)   # Makes image smaller for easier/speeding processing
    imgSmall = cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

    FacesCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall,FacesCurFrame)


    for encodeFace,faceLoc in zip(encodeCurFrame,FacesCurFrame):
        matches = face_recognition.compare_faces(knownEncodeList, encodeFace)
        faceDistance = face_recognition.face_distance(knownEncodeList, encodeFace)
        #print(faceDistance)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
#           print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            attendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1) & 0xFF  # Get the pressed key

    if key == ord('q'):  # Check if 'q' is pressed
        break

capture_video.release()
cv2.destroyAllWindows()


