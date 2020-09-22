import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
import time
import math

def get_neck_id(human):
    if 1 in human.body_parts.keys():#neck
        return 1
    else:
        return get_face_id(human)

def get_face_id(human):
    if 0 in human.body_parts.keys():#nose
        return 0
    elif 14 in human.body_parts.keys():#reye
        return 14
    elif 15 in human.body_parts.keys():#reye
        return 15
    else:
        return -1

def get_hip_id(human):
    if 8 in human.body_parts.keys():#rhip
        return 8
    elif 11 in human.body_parts.keys():#lhip
        return 11
    else:
        return -1

def get_knee_id(human):
    if 9 in human.body_parts.keys():#rknee
        return 9
    elif 12 in human.body_parts.keys():#lknee
        return 12
    else:
        return -1

def part_vertical_distance(part_id_1,part_id_2,human,image):
    #骨架图中的坐标为占比
    image_h, image_w = image.shape[:2]
    v_1 = int(human.body_parts[part_id_1].y * image_h + 0.5)
    v_2 = int(human.body_parts[part_id_2].y * image_h + 0.5)
    return abs(v_1-v_2)

def is_part_inface(human,face_id,face_location,image):
    body_part = human.body_parts[face_id]

    image_h, image_w = image.shape[:2]
    # 骨架图中的坐标为占比
    x_part = int(body_part.x * image_w + 0.5)
    #print('x_part:'+str(x_part))
    y_part = int(body_part.y * image_h + 0.5)
    #print('y_part:' + str(y_part))

    top, right, bottom, left = face_location
    #print('top:'+str(top),'right:'+str(right),'bottom:'+str(bottom),'left:'+str(left))
    if x_part < right and x_part > left and y_part > top and y_part < bottom:
        return True
    else:
        return False

def is_stand(human,image):
    thred = -0.97
    ##left_knee:12 right_knee:9
    ##left_hip:11  right_hip:
    ##neck:1       nose:0
    ##left_eye:15  right_eye:14
    point = []
    if 12 in human.body_parts.keys():
        body_part = human.body_parts[12]
        image_h, image_w = image.shape[:2]
        x_part = int(body_part.x * image_w + 0.5)
        y_part = int(body_part.y * image_h + 0.5)
        point.append([x_part, y_part])
        if 11 in human.body_parts.keys():
            body_part = human.body_parts[11]
            image_h, image_w = image.shape[:2]
            x_part = int(body_part.x * image_w + 0.5)
            y_part = int(body_part.y * image_h + 0.5)
            point.append([x_part, y_part])
            if 1 in human.body_parts.keys():
                body_part = human.body_parts[1]
                image_h, image_w = image.shape[:2]
                x_part = int(body_part.x * image_w + 0.5)
                y_part = int(body_part.y * image_h + 0.5)
                point.append([x_part, y_part])
                a = math.sqrt((point[0][0]-point[1][0])*(point[0][0]-point[1][0])+(point[0][1]-point[1][1])*(point[0][1]-point[1][1]))
                b = math.sqrt((point[1][0]-point[2][0])*(point[1][0]-point[2][0])+(point[1][1]-point[2][1])*(point[1][1]-point[2][1]))
                c = math.sqrt((point[0][0]-point[2][0])*(point[0][0]-point[2][0])+(point[0][1]-point[2][1])*(point[0][1]-point[2][1]))
                sita = (a*a+b*b-c*c)/(2*a*b)
                print("000000000000000000000000000000000000000000000")
                print("sita"+str(sita))
                print("000000000000000000000000000000000000000000000")
                return sita <= thred
            elif 0 in human.body_parts.keys():
                body_part = human.body_parts[0]
                image_h, image_w = image.shape[:2]
                x_part = int(body_part.x * image_w + 0.5)
                y_part = int(body_part.y * image_h + 0.5)
                point.append([x_part, y_part])
                a = math.sqrt(
                    (point[0][0] - point[1][0]) * (point[0][0] - point[1][0]) + (point[0][1] - point[1][1]) * (
                                point[0][1] - point[1][1]))
                b = math.sqrt(
                    (point[1][0] - point[2][0]) * (point[1][0] - point[2][0]) + (point[1][1] - point[2][1]) * (
                                point[1][1] - point[2][1]))
                c = math.sqrt(
                    (point[0][0] - point[2][0]) * (point[0][0] - point[2][0]) + (point[0][1] - point[2][1]) * (
                                point[0][1] - point[2][1]))
                sita = (a * a + b * b - c * c) / (2 * a * b)
                print("000000000000000000000000000000000000000000000")
                print("sita" + str(sita))
                print("000000000000000000000000000000000000000000000")
                return sita <= thred
            elif 15 in human.body_parts.keys():
                body_part = human.body_parts[15]
                image_h, image_w = image.shape[:2]
                x_part = int(body_part.x * image_w + 0.5)
                y_part = int(body_part.y * image_h + 0.5)
                point.append([x_part, y_part])
                a = math.sqrt(
                    (point[0][0] - point[1][0]) * (point[0][0] - point[1][0]) + (point[0][1] - point[1][1]) * (
                                point[0][1] - point[1][1]))
                b = math.sqrt(
                    (point[1][0] - point[2][0]) * (point[1][0] - point[2][0]) + (point[1][1] - point[2][1]) * (
                                point[1][1] - point[2][1]))
                c = math.sqrt(
                    (point[0][0] - point[2][0]) * (point[0][0] - point[2][0]) + (point[0][1] - point[2][1]) * (
                                point[0][1] - point[2][1]))
                sita = (a * a + b * b - c * c) / (2 * a * b)
                print("000000000000000000000000000000000000000000000")
                print("sita" + str(sita))
                print("000000000000000000000000000000000000000000000")
                return sita <= thred
    if 9 in human.body_parts.keys():
        body_part = human.body_parts[9]
        image_h, image_w = image.shape[:2]
        x_part = int(body_part.x * image_w + 0.5)
        y_part = int(body_part.y * image_h + 0.5)
        point.append([x_part, y_part])
        if 8 in human.body_parts.keys():
            body_part = human.body_parts[8]
            image_h, image_w = image.shape[:2]
            x_part = int(body_part.x * image_w + 0.5)
            y_part = int(body_part.y * image_h + 0.5)
            point.append([x_part, y_part])
            if 1 in human.body_parts.keys():
                body_part = human.body_parts[1]
                image_h, image_w = image.shape[:2]
                x_part = int(body_part.x * image_w + 0.5)
                y_part = int(body_part.y * image_h + 0.5)
                point.append([x_part, y_part])
                a = math.sqrt((point[0][0]-point[1][0])*(point[0][0]-point[1][0])+(point[0][1]-point[1][1])*(point[0][1]-point[1][1]))
                b = math.sqrt((point[1][0]-point[2][0])*(point[1][0]-point[2][0])+(point[1][1]-point[2][1])*(point[1][1]-point[2][1]))
                c = math.sqrt((point[0][0]-point[2][0])*(point[0][0]-point[2][0])+(point[0][1]-point[2][1])*(point[0][1]-point[2][1]))
                sita = (a*a+b*b-c*c)/(2*a*b)
                print("000000000000000000000000000000000000000000000")
                print("sita" + str(sita))
                print("000000000000000000000000000000000000000000000")
                return sita <= thred
            elif 0 in human.body_parts.keys():
                body_part = human.body_parts[0]
                image_h, image_w = image.shape[:2]
                x_part = int(body_part.x * image_w + 0.5)
                y_part = int(body_part.y * image_h + 0.5)
                point.append([x_part, y_part])
                a = math.sqrt(
                    (point[0][0] - point[1][0]) * (point[0][0] - point[1][0]) + (point[0][1] - point[1][1]) * (
                                point[0][1] - point[1][1]))
                b = math.sqrt(
                    (point[1][0] - point[2][0]) * (point[1][0] - point[2][0]) + (point[1][1] - point[2][1]) * (
                                point[1][1] - point[2][1]))
                c = math.sqrt(
                    (point[0][0] - point[2][0]) * (point[0][0] - point[2][0]) + (point[0][1] - point[2][1]) * (
                                point[0][1] - point[2][1]))
                sita = (a * a + b * b - c * c) / (2 * a * b)
                print("000000000000000000000000000000000000000000000")
                print("sita" + str(sita))
                print("000000000000000000000000000000000000000000000")
                return sita <= thred
            elif 14 in human.body_parts.keys():
                body_part = human.body_parts[14]
                image_h, image_w = image.shape[:2]
                x_part = int(body_part.x * image_w + 0.5)
                y_part = int(body_part.y * image_h + 0.5)
                point.append([x_part, y_part])
                a = math.sqrt(
                    (point[0][0] - point[1][0]) * (point[0][0] - point[1][0]) + (point[0][1] - point[1][1]) * (
                                point[0][1] - point[1][1]))
                b = math.sqrt(
                    (point[1][0] - point[2][0]) * (point[1][0] - point[2][0]) + (point[1][1] - point[2][1]) * (
                                point[1][1] - point[2][1]))
                c = math.sqrt(
                    (point[0][0] - point[2][0]) * (point[0][0] - point[2][0]) + (point[0][1] - point[2][1]) * (
                                point[0][1] - point[2][1]))
                sita = (a * a + b * b - c * c) / (2 * a * b)
                print("000000000000000000000000000000000000000000000")
                print("sita" + str(sita))
                print("000000000000000000000000000000000000000000000")
                return sita <= thred
    return True

def is_stand1(human,image):
    neck_id = get_neck_id(human)
    hip_id = get_hip_id(human)
    knee_id = get_knee_id(human)
    if hip_id > 0 and knee_id > 0:  # hip和knee均检测到
        neck_hip_d = part_vertical_distance(neck_id, hip_id, human, image)
        hip_knee_d = part_vertical_distance(hip_id, knee_id, human, image)
        # minus = abs(neck_hip_d-hip_knee_d)
        per = hip_knee_d / neck_hip_d
        print('hip_knee_d / neck_hip_d: ' + str(per))
        return per > 0.6
    else:  # 存在遮挡即认为站着
        return True

def get_pose_model(image):
    print('e-model-begin')
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
    print('e-model-built')
    humans = e.inference(image, resize_to_default=False, upsample_size=4.0)
    #骨架图显示，会改变原图
    #image1 = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    #cv2.imshow('rect_1',image1)
    return humans

def pose_predict(humans,face_location,image):
    #遍历所有骨架
    for human in humans:
        #找到该骨架的人脸的位置(各部位的id对应查看./reference/common.py)
        face_id = get_face_id(human)
        #print(face_id)
        if face_id >= 0:
            #判断该骨架的人脸在不在当前人脸位置中
            if is_part_inface(human,face_id,face_location,image):
                if is_stand1(human,image):
                    return 'stand'
                else:
                    return 'sit'
    return 'notfind'

def test(image):
    e = TfPoseEstimator(get_graph_path('cmu'), target_size=(432, 368))
    humans = e.inference(image, resize_to_default=False, upsample_size=4.0)
    #image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    for human in humans:
        if is_stand(human, image):
            return 'stand'
        else:
            return 'sit'

if __name__ == '__main__':

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    #time_begin =  time.localtime()
    image = cv2.imread('./0000_color.jpg')

    cv2.putText(image,pose, (30,30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 1)
    cv2.imshow("rect", image)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    #time_end = time.localtime()
    #print(time.strftime("time_begin: %Y-%m-%d %H:%M:%S",time_begin,"time_end: %Y-%m-%d %H:%M:%S",time_end))
    key = cv2.waitKey(-1)
    if key == 27:  # ESC
        cv2.destroyAllWindows()












