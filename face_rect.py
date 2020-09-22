
import cv2

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
import os
import keras
import tensorflow as tf
import numpy as np

def get_pose_model(image):
    print('e-model-begin')
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
    print('e-model-built')
    humans = e.inference(image, resize_to_default=False, upsample_size=4.0)
    return humans

def face_rect_pose(image,humans):
    #humans = get_pose_model(image)
    print(humans.__len__())
    faces = []
    for human in humans:
        face = []
        if 0 in human.body_parts.keys():  # nose
            x1_part = 0
            y1_part = 0
            body_part = human.body_parts[0]
            image_h, image_w = image.shape[:2]
            # 骨架图中的坐标为占比
            x_part = int(body_part.x * image_w + 0.5)
            # print('x_part:'+str(x_part))
            y_part = int(body_part.y * image_h + 0.5)
            face.append([x_part, y_part])
            # print('y_part:' + str(y_part))
            if 1 in human.body_parts.keys():
                body_part = human.body_parts[1]
                image_h, image_w = image.shape[:2]
                # 骨架图中的坐标为占比
                x1_part = int(body_part.x * image_w + 0.5)
                # print('x_part:'+str(x_part))
                y1_part = int(body_part.y * image_h + 0.5)
                # print('y_part:' + str(y_part))
                face.append([x_part, y_part + (y1_part - y_part) // 2])
            else:
                if 14 in human.body_parts.keys():
                    body_part = human.body_parts[14]
                    image_h, image_w = image.shape[:2]
                    y2_part = int(body_part.y * image_h + 0.5)
                    face.append([x_part, y_part + (y2_part - y_part)])
        if 6 in human.body_parts.keys():
            body_part = human.body_parts[6]
            image_h, image_w = image.shape[:2]
            # 骨架图中的坐标为占比
            x_part = int(body_part.x * image_w + 0.5)
            # print('x_part:'+str(x_part))
            y_part = int(body_part.y * image_h + 0.5)
            # print('y_part:' + str(y_part))
        #    face.append([x_part, y_part])
        if 14 in human.body_parts.keys():
            body_part = human.body_parts[14]
            image_h, image_w = image.shape[:2]
            # 骨架图中的坐标为占比
            x_part = int(body_part.x * image_w + 0.5)
            # print('x_part:'+str(x_part))
            y_part = int(body_part.y * image_h + 0.5)
            # print('y_part:' + str(y_part))
            face.append([x_part, y_part])
        if 15 in human.body_parts.keys():
            body_part = human.body_parts[15]
            image_h, image_w = image.shape[:2]
            # 骨架图中的坐标为占比
            x_part = int(body_part.x * image_w + 0.5)
            # print('x_part:'+str(x_part))
            y_part = int(body_part.y * image_h + 0.5)
            # print('y_part:' + str(y_part))
            face.append([x_part, y_part])
        if 16 in human.body_parts.keys():
            body_part = human.body_parts[16]
            image_h, image_w = image.shape[:2]
            # 骨架图中的坐标为占比
            x_part = int(body_part.x * image_w + 0.5)
            # print('x_part:'+str(x_part))
            y_part = int(body_part.y * image_h + 0.5)
            # print('y_part:' + str(y_part))
            face.append([x_part, y_part])
        if 17 in human.body_parts.keys():
            body_part = human.body_parts[17]
            image_h, image_w = image.shape[:2]
            # 骨架图中的坐标为占比
            x_part = int(body_part.x * image_w + 0.5)
            # print('x_part:'+str(x_part))
            y_part = int(body_part.y * image_h + 0.5)
            # print('y_part:' + str(y_part))
            face.append([x_part, y_part])
        print(face)
        if (face.__len__() == 6):
            cnt = np.array(face)  # 必须是array数组的形式
            rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            print(rect)
            box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点坐标
            print(box)
            box = np.int0(box)
            print(box)
            minx = 10000
            miny = 10000
            maxx = 0
            maxy = 0
            for i in range(4):
                if (box[i][0] < minx):
                    minx = box[i][0]
                if (box[i][1] < miny):
                    miny = box[i][1]
                if (box[i][0] > maxx):
                    maxx = box[i][0]
                if (box[i][1] > maxy):
                    maxy = box[i][1]
            # miny = miny - (maxy-miny)
            # maxy = maxy + (maxy-miny)
            #img = cv2.rectangle(image, (minx, miny), (maxx, maxy), (0, 255, 0), 1)
            faces.append([minx,miny,maxx,maxy])
    return faces


if __name__ == '__main__':
    image_dir = './dataset/'
    i = 0
    image = cv2.imread('test.png')
    humans = get_pose_model(image)
    faces = face_rect_pose(image,humans)
    print(faces)
    for i in range(faces.__len__()):
        img = cv2.rectangle(image,(faces[i][0],faces[i][1]),(faces[i][2],faces[i][3]),(0,255,0),1)
    cv2.imshow("11",img)
    cv2.waitKey(0)
    '''
    for image_path in os.listdir(image_dir):
        print(i)
        i=i+1
        image = cv2.imread(image_dir + image_path)
        image = cv2.imread('./photo/0000_color.jpg')
        humans = get_pose_model(image)
        print(humans.__len__())
        faces = []
        for human in humans:
            face = []
            if 0 in human.body_parts.keys():  # nose
                x1_part = 0
                y1_part = 0
                body_part = human.body_parts[0]
                image_h, image_w = image.shape[:2]
                # 骨架图中的坐标为占比
                x_part = int(body_part.x * image_w + 0.5)
                # print('x_part:'+str(x_part))
                y_part = int(body_part.y * image_h + 0.5)
                face.append([x_part, y_part])
                # print('y_part:' + str(y_part))
                if 1 in human.body_parts.keys():
                    body_part = human.body_parts[1]
                    image_h, image_w = image.shape[:2]
                    # 骨架图中的坐标为占比
                    x1_part = int(body_part.x * image_w + 0.5)
                    # print('x_part:'+str(x_part))
                    y1_part = int(body_part.y * image_h + 0.5)
                    # print('y_part:' + str(y_part))
                    face.append([x_part, y_part+(y1_part-y_part)//2])
                else:
                    if 14 in human.body_parts.keys():
                        body_part = human.body_parts[14]
                        image_h, image_w = image.shape[:2]
                        y2_part = int(body_part.y * image_h + 0.5)
                        face.append([x_part,y_part+(y2_part-y_part)])
            if 1 in human.body_parts.keys():
                body_part = human.body_parts[1]
                image_h, image_w = image.shape[:2]
                # 骨架图中的坐标为占比
                x_part = int(body_part.x * image_w + 0.5)
                # print('x_part:'+str(x_part))
                y_part = int(body_part.y * image_h + 0.5)
                # print('y_part:' + str(y_part))
                #face.append([x_part, y_part])
            if 6 in human.body_parts.keys():
                body_part = human.body_parts[6]
                image_h, image_w = image.shape[:2]
                # 骨架图中的坐标为占比
                x_part = int(body_part.x * image_w + 0.5)
                # print('x_part:'+str(x_part))
                y_part = int(body_part.y * image_h + 0.5)
                # print('y_part:' + str(y_part))
            #    face.append([x_part, y_part])
            if 14 in human.body_parts.keys():
                body_part = human.body_parts[14]
                image_h, image_w = image.shape[:2]
                # 骨架图中的坐标为占比
                x_part = int(body_part.x * image_w + 0.5)
                # print('x_part:'+str(x_part))
                y_part = int(body_part.y * image_h + 0.5)
                # print('y_part:' + str(y_part))
                face.append([x_part, y_part])
            if 15 in human.body_parts.keys():
                body_part = human.body_parts[15]
                image_h, image_w = image.shape[:2]
                # 骨架图中的坐标为占比
                x_part = int(body_part.x * image_w + 0.5)
                # print('x_part:'+str(x_part))
                y_part = int(body_part.y * image_h + 0.5)
                # print('y_part:' + str(y_part))
                face.append([x_part, y_part])
            if 16 in human.body_parts.keys():
                body_part = human.body_parts[16]
                image_h, image_w = image.shape[:2]
                # 骨架图中的坐标为占比
                x_part = int(body_part.x * image_w + 0.5)
                # print('x_part:'+str(x_part))
                y_part = int(body_part.y * image_h + 0.5)
                # print('y_part:' + str(y_part))
                face.append([x_part, y_part])
            if 17 in human.body_parts.keys():
                body_part = human.body_parts[17]
                image_h, image_w = image.shape[:2]
                # 骨架图中的坐标为占比
                x_part = int(body_part.x * image_w + 0.5)
                # print('x_part:'+str(x_part))
                y_part = int(body_part.y * image_h + 0.5)
                # print('y_part:' + str(y_part))
                face.append([x_part, y_part])
            print(face)
            if(face.__len__() == 6):
                cnt = np.array(face)  # 必须是array数组的形式
                rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
                print(rect)
                box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点坐标
                print(box)
                box = np.int0(box)
                print(box)
                minx = 10000
                miny = 10000
                maxx = 0
                maxy = 0
                for i in range(4):
                    if(box[i][0] < minx):
                        minx = box[i][0]
                    if(box[i][1] < miny):
                        miny = box[i][1]
                    if(box[i][0] > maxx):
                        maxx = box[i][0]
                    if(box[i][1] > maxy):
                        maxy = box[i][1]
                #miny = miny - (maxy-miny)
                #maxy = maxy + (maxy-miny)
                img = cv2.rectangle(image, (minx,miny),(maxx,maxy), (0, 255, 0), 1)
        cv2.imshow('1', img)
        cv2.waitKey(0)
        break
    '''
    












