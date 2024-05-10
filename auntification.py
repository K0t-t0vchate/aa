import time

import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['Unknown','Irina','Ilya','Artemiy','Zahar','Seryi','Lavrenty']

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)



while True:
    rs = input('\n зарегистрируйтесь или войдите рег вход  ')
    if rs == 'рег':
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)

        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        face_id = input('\n введи юзер айди ')

        print('\n Установка распознования. Смотрите четко в камеру и ожидайте..')
        count = 0

        while (True):

            ret, img = cam.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1

                cv2.imwrite('face/user.' + str(face_id) + '.' + str(count) + '.jpg', gray[y:y + h, x:x + w])

                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break
            elif count >= 10:
                break

        print('\n Выход из программы и зачистка данных')
        cam.release()
        cv2.destroyAllWindows()
        break

    elif rs == 'вход':
        while True:
            ret, img = cam.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

                if (confidence < 70):
                    id = names[id]
                    print('добро пожаловать', confidence, id)
                    confidence = ' {0}%'.format(round(100 - confidence))

                else:
                    id = 'unknown'
                    print('зарегайся, а потом уже лицом свети!')
                    confidence = ' {0}%'.format(round(100 - confidence))


                cv2.putText(img, str(id), (x+5,y-5), font, 1, (0,0,0), 4)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
                break


        cv2.imshow('camera', img)


    print(23)
    if id=='Ilya':
        time.sleep(2)
        break

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
print('\n Выход из программы и чистка данных..')
cam.release()
cv2.destroyAllWindows()