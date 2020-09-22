import cv2
import numpy as np
from wide_resnet import WideResNet
import face_recognition

def gender_predict(face_location,img):
    depth = 16
    k = 8
    img_size = 64
    margin = 0.4

    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)

    weight_file = './pretrained_models/weights.28-3.73.hdf5'
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    #x1, y1, x2, y2, w, h = face_location.left(), face_location.top(), face_location.right() + 1, face_location.bottom() + 1, face_location.width(), face_location.height()
    #left, top, right, bottom = face_location.left(), face_location.top(), face_location.right() + 1, face_location.bottom()

    top, right, bottom, left = face_location
    w = right - left
    h = bottom - top
    xw1 = max(int(left - margin * w), 0)
    yw1 = max(int(top - margin * h), 0)
    xw2 = min(int(right + margin * w), img_w - 1)
    yw2 = min(int(bottom + margin * h), img_h - 1)

    #取下待检测人脸
    face = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
    face = np.expand_dims(face, 0)

    #传入代检测人脸
    results = model.predict(face)
    #print(results[0][0][0])

    if results[0][0][0] < 0.5:
        label = 'male'
    else:
        label = 'female'

    return label
'''
img_people = cv2.imread('./dataset/0014_color.jpg')
img_people_location = face_recognition.face_locations(img_people)

label = gender_predict(img_people_location[0],img_people)
print(label)
'''