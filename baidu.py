# -*- coding: utf-8 -*-
__author__ = 'fff_zrx'
import requests
import base64
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

def cv2_base64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str

def get_face_result(cv2_img):
    # 获取access_token
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=1WHQrZiBNubCSxYfdBTgev1j&client_secret=YnI7nVIIFa5wG2z0lonkgTGamVCuvI1Q'
    header = {'Content-Type': 'application/json; charset=UTF-8'}
    response1 = requests.post(url=host, headers=header)  # <class 'requests.models.Response'>
    json1 = response1.json()  # <class 'dict'>
    access_token = json1['access_token']

    # 转换图片格式
    base64_img = cv2_base64(cv2_img)

    # 访问人脸检测api
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"
    params = {"image": base64_img, "image_type": "BASE64", "face_field": "age,gender,face_shape,", "max_face_num": "10"}
    header = {'Content-Type': 'application/json'}
    request_url = request_url + "?access_token=" + access_token
    response1 = requests.post(url=request_url, data=params, headers=header)  # <class 'requests.models.Response'>
    face_result = response1.json()['result']
    #print(face_result)
    return face_result

def get_face_locations_b(face_result):
    #face_result = get_face_result(cv2_img)
    face_num = face_result['face_num']
    face_list = face_result['face_list']

    face_locations = []
    face_location = []
    for i in range (face_num):
        face_loaction_dic = face_list[i]['location']
        #top, right, bottom, left
        face_location.append(int(face_loaction_dic['top']))
        face_location.append(int(face_loaction_dic['left'] + face_loaction_dic['width']))
        face_location.append(int(face_loaction_dic['top'] + face_loaction_dic['height']))
        face_location.append(int(face_loaction_dic['left']))
        face_locations.append((tuple(face_location)))
        face_location.clear()

    #print(face_locations)
    return face_locations

def get_gender_list(face_result):
    #face_result = get_face_result(cv2_img)
    face_num = face_result['face_num']
    face_list = face_result['face_list']


    gender_list = []
    for i in range(face_num):
        gender_dic = face_list[i]['gender']
        gender_list.append(gender_dic['type'])

    print(gender_list)
    return gender_list

def b_deal(cv2_img,selection):
    unknow_img = cv2.resize(cv2_img,(1920, 1080))
    result = get_face_result(cv2_img)
    face_locations = get_face_locations_b(result)
    gender_list = get_gender_list(result)
    face_num = len(face_locations)
    if(face_num>0):
        for i in range(face_num):
            top, right, bottom, left = face_locations[i]
            gender = gender_list[i]
            draw_image = cv2.rectangle(unknow_img, (left, top), (right, bottom), (0, 0, 255), 2)
            # 在图片上写标号，性别
            if(selection == True):
                cv2.putText(draw_image,str(i+1)+' '+gender, (left, bottom), cv2.FONT_HERSHEY_TRIPLEX, 0.6,(0, 255, 0), 1)
        return draw_image,face_num
    return False,0


if __name__ == '__main__':

    cv2_img = cv2.imread('test.png')
    result = get_face_result(cv2_img)
    get_gender_list(result)
    #cv2.imshow('test',draw_image)
    #print(face_num)
    #key = cv2.waitKey(-1)
    #if key == 27:  # ESC
     #   cv2.destroyAllWindows()




