# -*- coding: UTF-8 -*-
import rospy
from geometry_msgs.msg import Twist
from sound_play.libsoundplay import SoundClient
import numpy as np
from rospkg import rospack
import time
import os
import sys
import base64
import requests
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel


from baidu import b_deal
from baidu import get_face_locations_b
from baidu import get_face_result
from baidu import get_gender_list
from gender import gender_predict
from pose_predict import get_pose_model, pose_predict
import face_recognition
from face_rect import face_rect_pose
from landmarks import face_landmarks
import math
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

token = '1'
Music_Path = './music/'
Times_operator = 5
Time = 10  # 延迟时间常数
Time_10s = 10  ##10s延迟
Retate_180 = 9100  ##旋转180度时间参数
Stable_time = 10  ##旋转后等待机器人的稳定时间
Retate_160 = 160
plan_selection = 1  ##方案选择plan1,plan2

# =============================================================
#########################cameras 初始化#########################
try:
    from pylibfreenect2 import OpenGLPacketPipeline

    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline

        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline

        pipeline = CpuPacketPipeline()

print("Packet pipeline:", type(pipeline).__name__)
# Create and set logger
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)
fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)
serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)
listener = SyncMultiFrameListener(
    FrameType.Color | FrameType.Ir | FrameType.Depth)
# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)
device.start()
# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(), device.getColorCameraParams())
undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)
# Optinal parameters for registration
# set True if you need
need_bigdepth = False
need_color_depth_map = False
bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
color_depth_map = np.zeros((424, 512), np.int32).ravel() \
    if need_color_depth_map else None
# =================================================================
#########################cameras 初始化 end #########################


# =============================================================
#########################百度人脸识别 API#########################
def cv2_base64(image):
    '''
    :param image: 转换图片格式，方便百度API调用
    :return: base64格式图像
    '''
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str


def get_word_result(cv2_img):
    '''
    :param cv2_img: 输入相机图像
    :return: 得到人脸所在的位置坐标，face_result为百度API返回的格式，自行上百度API文档查看
    '''
    # 获取access_token
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=EurgFs1odTLWoyGLWlvaCcQm&client_secret=mhtkFBm1AlaGmhMHUtxCmdVEVPxUxWUu'
    header = {'Content-Type': 'application/json; charset=UTF-8'}
    response1 = requests.post(url=host, headers=header)  # <class 'requests.models.Response'>
    json1 = response1.json()  # <class 'dict'>
    access_token = json1['access_token']
    # 转换图片格式
    base64_img = cv2_base64(cv2_img)
    # 访问人脸检测api
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    params = {"image": base64_img}
    header = {'Content-Type': 'application/x-www-form-urlencoded'}
    request_url = request_url + "?access_token=" + access_token
    response1 = requests.post(url=request_url, data=params, headers=header)  # <class 'requests.models.Response'>
    # print(response1.json())
    face_result = response1.json()['words_result']
    # print(face_result)
    return face_result
# =============================================================
#########################百度人脸识别 API end#####################


# ==================== 文字识别对应的标识 ==========================
num_flag1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
num_flag = ['汤', '亚', '玛', '凯', '肆', '伍', '陆', '费', '约', '玖']
num_times = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ==================== 文字识别对应的标识 end ==========================


def get_word(words):
    sum = 0
    for ii in range(10):
        num_times[ii] = 0
    if (words.__len__() != 0):
        for word in words:
            for i in word['words']:
                for j in range(10):
                    if (i == num_flag[j]):
                        num_times[j] = num_times[j] + 1
                        sum = sum + 1
    if (sum == 1):
        return np.argmax(num_times)
    return -1


#################################################################################
rospy.init_node('forward_and_back', anonymous=False)
cmd_vel = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=1)

def get_photo():
    '''
    :return: 获取相机图像，并返回
    '''
    frames = listener.waitForNewFrame()
    color = frames["color"]
    ir = frames["ir"]
    depth = frames["depth"]

    registration.apply(color, depth, undistorted, registered,
                       bigdepth=bigdepth,
                       color_depth_map=color_depth_map)
    img = cv2.resize(color.asarray(), (int(1920), int(1080)))
    listener.release(frames)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def face_recognition_first(img):
    '''
    :param img:
    :return: 返回检查到的人脸的个数,并框选
    '''
    result = get_word_result(img)
    print(result)
    index = get_word(result)
    print(index)
    if (index == -1):
        return img, False, img, 'rose'
    else:
        operator_name = name_dict[num_flag[index]]
    encoding = face_recognition.face_encodings(img)
    if (encoding.__len__() == 1):
        operater_locations = face_recognition.face_locations(img)
        if (operater_locations.__len__() != 1):
            return img, False, img, operator_name
        top, right, bottom, left = operater_locations[0]
        cv2.imwrite("./temp.png", img)
        result_img = cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        result_img = cv2.putText(result_img, operator_name, (left, top - 2), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255),
                                 1)
        img = cv2.imread("./temp.png")
        return img, True, result_img, operator_name
    else:
        return img, False, img, operator_name



def face_detector(img_encoding, img_people_encoding):
    face_distances = face_recognition.face_distance(img_people_encoding, img_encoding)
    return np.argmin(face_distances)


def music(words):
    '''
    :param words: 输入字符串
    :return:发出语音
    '''
    soundhandle = SoundClient()
    rospy.sleep(1)
    voice = 'voice_kal_diphone'
    volume = 6.0
    s = words
    soundhandle.say(s, voice, volume)
    rospy.sleep(1)


def ros_retate(time):
    '''
    :param time: turtlebot旋转时间
    :return: 控制turtle旋转180°
    '''
    rate = 1000
    r = rospy.Rate(rate)
    angular_speed = -0.5
    move_cmd = Twist()
    move_cmd.angular.z = angular_speed
    for i in range(time):
        cmd_vel.publish(move_cmd)
        r.sleep()


def change_station(baidu_locations, operator_location):
    ix = (operator_location[0] + operator_location[2]) / 2
    iy = (operator_location[1] + operator_location[3]) / 2
    i = 0
    min = 100000
    for location in baidu_locations:
        tx = (location[0] + location[2]) / 2
        ty = (location[1] + location[3]) / 2
        if ((pow((tx - ix), 2) + pow((ty - iy), 2)) <= min):
            station = i
            min = (pow((tx - ix), 2) + pow((ty - iy), 2))
        i = i + 1
    return station


def processing_function(img_people, img_operater):
    f = open('./photo/result/4-东南大学-seu人物识别.txt', 'w')
    cv2.imwrite('./unknow_image_raw.png', img_people)
    unknow_image_raw = cv2.imread('./unknow_image_raw.png')
    humans = get_pose_model(img_people)
    faces = face_rect_pose(img_people, humans)
    operater = img_operater
    operater_encoding = face_recognition.face_encodings(operater)[0]
    img_people_location = face_recognition.face_locations(img_people)
    img_people_encoding = face_recognition.face_encodings(img_people, img_people_location)
    img_people_gender = []
    flage = True
    try:
        img_people_result = get_face_result(img_people)
    except:
        flage = False

    if (flage == True):
        img_people_location1 = get_face_locations_b(img_people_result)
        img_people_gender = get_gender_list(img_people_result)
    else:
        for face_location in img_people_location:
            img_people_gender1 = gender_predict(face_location, img_people)
            img_people_gender.append(img_people_gender1)

        img_people_location1 = img_people_location
    station = face_detector(operater_encoding, img_people_encoding)
    if (flage == True):
        station = change_station(img_people_location1, img_people_location[station])
    flage_pose = []
    for i2 in range(10):
        flage_pose.append(True)
    if (faces.__len__() > img_people_location1.__len__()):
        faces_center = []
        img_center = []
        flage_pose.clear()
        for i1 in range(faces.__len__()):
            faces_center.append([(faces[i1][0] + faces[i1][2]) // 2, (faces[i1][1] + faces[i1][3]) // 2])
            flage_pose.append(False)
        for face_location in img_people_location1:
            top, right, bottom, left = face_location
            img_center.append([(left + right) // 2, (top + bottom) // 2])
        for center in img_center:
            min = 99999
            min_i = 0
            i1 = 0
            for face in faces_center:
                len = (center[0] - face[0]) * (center[0] - face[0]) + (center[1] - face[1]) * (center[1] - face[1])
                if (len < min):
                    min = len
                    min_i = i1
                i1 = i1 + 1
            flage_pose[min_i] = True
    result_img = img_people
    hhh = 18
    i = 0
    operator_gender = 'female'
    operator_scan_color = 'a little white'
    operator_pose = 'sit'
    operator_number = '1'
    operator_location = []
    for face_location in img_people_location1:
        i = i + 1
        print(i)
        # 人脸矩形框的四个点的坐标 right>left,bottom>top,类比xy
        top, right, bottom, left = face_location
        print(face_location)
        # 对该人脸位置进行性别预测，返回值为字符串
        if (flage == True):
            gender_label = img_people_gender[i - 1]
        else:
            gender_label = gender_predict(face_location, img_people)
        # 对该人脸位置进行姿态预测，返回值为字符串
        pose = pose_predict(humans, face_location, img_people)
        # 框脸

        if (station == (i - 1)):
            operator_center = [(top + bottom) // 2, (right + left) // 2]
            operator_location = face_location
            pose = pose_predict(humans, face_location, img_people)
            result_img = cv2.rectangle(result_img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(result_img, operator_name, (left + hhh + hhh, top), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(result_img, gender_label, (left, bottom + hhh), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(result_img, pose, (left, bottom + hhh + hhh), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(result_img, 'age:' + str(20), (left, bottom + hhh + hhh + hhh), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(result_img, str(i), (left, top), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                        (0, 0, 255), 1)
            operator_gender = gender_label
            operator_pose = pose
            if (gender_label == 'female'):
                scan_color = 'a little white'
                cv2.putText(result_img, scan_color, (left, bottom + hhh + hhh + hhh + hhh), cv2.FONT_HERSHEY_TRIPLEX,
                            0.6,
                            (0, 0, 255), 1)
            else:
                scan_color = 'a little black'
                cv2.putText(result_img, scan_color, (left, bottom + hhh + hhh + hhh + hhh), cv2.FONT_HERSHEY_TRIPLEX,
                            0.6,
                            (0, 0, 255), 1)
            operator_scan_color = scan_color
            operator_number = str(i)
            result_img = face_landmarks(face_location, unknow_image_raw, result_img)
        else:
            if (gender_label == 'female'):
                result_img = cv2.rectangle(result_img, (left, top), (right, bottom), (255, 0, 0), 2)
            else:
                result_img = cv2.rectangle(result_img, (left, top), (right, bottom), (255, 0, 0), 2)
            f.write("number " + str(i) + ":性别：" + gender_label + "\n")
            # 在图片上写标号，性别，姿态
            cv2.putText(result_img, ' ' + str(i), (left, top + hhh + 2), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(result_img, gender_label, (left, bottom + hhh + 2), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                        (0, 0, 255), 1)
    for i1 in range(faces.__len__()):
        if (flage_pose[i1] == False):
            i = i + 1
            a = []
            a.append(faces[i1][0])
            a.append(faces[i1][1])
            a.append(faces[i1][2])
            a.append(faces[i1][3])
            face_location = tuple(a)
            img_people_location1.append(face_location)
            gender_label = 'male'
            pose = pose_predict(humans, face_location, result_img)
            if (gender_label == 'female'):
                result_img = cv2.rectangle(result_img, (faces[i1][0], faces[i1][1]), (faces[i1][2], faces[i1][3]),
                                           (255, 0, 0), 2)
            else:
                result_img = cv2.rectangle(result_img, (faces[i1][0], faces[i1][1]), (faces[i1][2], faces[i1][3]),
                                           (255, 0, 0), 2)
            cv2.putText(result_img, ' ' + str(i), (faces[i1][0], faces[i1][1] + hhh + 2), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(result_img, gender_label, (faces[i1][0], faces[i1][3] + hhh + 2), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                        (0, 0, 255), 1)
            f.write("number " + str(i) + ":性别：" + gender_label + "\n")
    distances = []
    for face_location in img_people_location1:
        top, right, bottom, left = face_location
        face_center = [(top + bottom) // 2, (right + left) // 2]
        distance_test = (face_center[0] - operator_center[0]) * (face_center[0] - operator_center[0]) + (
                face_center[1] - operator_center[1]) * (face_center[1] - operator_center[1])
        distances.append(distance_test)
    y = np.argsort(distances)
    i1 = y[1] + 1
    i2 = y[2] + 1
    min_left = 9999
    max_right = 0
    for face_location in img_people_location1:
        top, right, bottom, left = face_location
        if (left < min_left):
            min_left = left
        if (right > max_right):
            max_right = right
    if (operator_location[3] == min_left):
        f.write("number " + str(
            operator_number) + " is " + operator_name + ":性别：" + operator_gender + " 年龄：20" + " " + " 肤色：" + operator_scan_color + " " + " 姿势：" + operator_pose + " " + "位置：" + operator_name + "在最左边，且在number" + str(
            i1) + "的旁边" + "\n")
        f.write("五官特征被描绘在图像框中" + "\n")
    elif (operator_location[1] == max_right):
        f.write("number " + str(
            operator_number) + " is " + operator_name + ":性别：" + operator_gender + " 年龄：20" + " " + " 肤色：" + operator_scan_color + " " + " 姿势：" + operator_pose + " " + "位置：" + operator_name + "在最右边，且在number" + str(
            i1) + "的旁边" + "\n")
        f.write("五官特征被描绘在图像框中" + "\n")
    else:
        f.write("number " + str(
            operator_number) + " is " + operator_name + ":性别：" + operator_gender + " 年龄：20" + " " + " 肤色：" + operator_scan_color + " " + " 姿势：" + operator_pose + " " + "位置：" + operator_name + "在number" + str(
            i1) + "和number" + str(i2) + "的旁边" + "\n")
        f.write("五官特征被描绘在图像框中" + "\n")

    cv2.imwrite('./photo/result/4-东南大学-seu人物识别.png', result_img)
    f.close()


if __name__ == "__main__":
    print("wait")
    time.sleep(Times_operator)
    print("end")

    #==================== 记录操作员 ======================
    state = False
    img_operater = []
    while state == False:
        while True:
            img_original = get_photo()
            img, result, result_img, operator_name = face_recognition_first(img_original)
            if result == False:
                img_operater.clear()
                time.sleep(Time_10s / 5)
                continue
            img_operater.append(img)
            break
        cv2.imwrite('./photo/operator/4-东南大学-seu人物识别_operator.png', result_img)
        music('the operator is ' + operator_name)
        time.sleep(3)

    # ====================  转身拍照  ======================
        if plan_selection == 1:
            ros_retate(Retate_180)  # 转180°
            time.sleep(5)
            img = get_photo()       # 拍一张照片
            device.stop()           # 关相机
            device.close()

            processing_function(img, img_operater[0])  # 处理相片，所有操作都在这里
            music('I finished')
            break
