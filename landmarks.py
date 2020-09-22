from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np

def is_point_inface(face_location,point):


    x_part = point[0]
    #print('x_part:'+str(x_part))
    y_part = point[1]
    #print('y_part:' + str(y_part))

    top, right, bottom, left = face_location
    #print('top:'+str(top),'right:'+str(right),'bottom:'+str(bottom),'left:'+str(left))
    if x_part < right and x_part > left and y_part > top and y_part < bottom:
        return True
    else:
        return False

'''
# Load the jpg file into a numpy array
image = face_recognition.load_image_file("./dataset/0011_color.jpg")
image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_CUBIC)
#image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_CUBIC)

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

# Create a PIL imagedraw object so we can draw on the picture
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:

    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=1)

# Show the picture
pil_image.show()


def face_landmarks(face_location,image):
    margin = 0.4
    img_size = 64
    img_h, img_w, _ = np.shape(image)
    top, right, bottom, left = face_location
    w = right - left
    h = bottom - top
    xw1 = max(int(left - margin * w), 0)
    yw1 = max(int(top - margin * h), 0)
    xw2 = min(int(right + margin * w), img_w - 1)
    yw2 = min(int(bottom + margin * h), img_h - 1)

    # 取下待检测人脸
    face = cv2.resize(image[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
    face_landmarks_list = face_recognition.face_landmarks(face)

    print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    # Create a PIL imagedraw object so we can draw on the picture
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        for facial_feature in face_landmarks.keys():
            print("The {} in this face has the following points: {}".format(facial_feature,
                                                                            face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=1)
    return pil_image

    # Show the picture
'''
def face_landmarks(face_location,image_raw,image_draw):

    face_landmarks_list = face_recognition.face_landmarks(image_raw)

    pil_image = Image.fromarray(image_draw)
    d = ImageDraw.Draw(pil_image)
    flag = False
    for face_landmarks in face_landmarks_list:

        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            if(is_point_inface(face_location,face_landmarks[facial_feature][0])):
                d.line(face_landmarks[facial_feature], width=1)
                flag = True
        if flag:
            cv2_img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_BGR2RGB)
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            return cv2_img
    cv2_img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_BGR2RGB)
    return cv2_img


if __name__ == '__main__':
    image = face_recognition.load_image_file("./dataset/0011_color.jpg")
    image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    face_locations = face_recognition.face_locations(image)
    print(face_locations)
    draw_image = image
    # 如果有人脸，（单张人脸图不可调试
    if (len(face_locations) > 0):
        i = 0
        for face_location in face_locations:
            top, right, bottom, left = face_location
            i = i + 1
            draw_image = cv2.putText(draw_image, "operator", (left, top), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 1)
            draw_image = face_landmarks(face_location,image,draw_image)
            cv2.imshow('landmarks',draw_image)
            key = cv2.waitKey(-1)
            if key == 27:  # ESC
                cv2.destroyAllWindows()



            #cv2.putText(image, str(i), (left, bottom), cv2.FONT_HERSHEY_TRIPLEX, 0.8,(0, 255, 0), 1)



