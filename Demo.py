import os
import face_recognition
from PIL import ImageGrab
import cv2
import numpy as np

# 目标人物
targer = ["Targer", "targer"]

# 人物图片文件夹
face_databases_dir = 'face_databases'
# 储存人物姓名
user_names = []
# 储存人物面部特征向量
user_faces_encodings = []
# 使用os得到人物图片文件夹下所有文件名
files = os.listdir(face_databases_dir)
# 使用循环取出文件名
for image_shot_name in files:
    # 截取文件名.前面的部分存入user_names中
    user_name, _ = os.path.splitext(image_shot_name)
    user_names.append(user_name)

    # 读取图片文件中的面部特征信息存入user_faces_encodings中
    image_shot_name = os.path.join(face_databases_dir, image_shot_name)
    image_file = face_recognition.load_image_file(image_shot_name)
    face_encoding = face_recognition.face_encodings(image_file)[0]
    user_faces_encodings.append(face_encoding)

# 使用ImageGrab.grab()循环获取当前屏幕截图并使用cv2.imshow()显示
while True:
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Video', 960, 540)
    # 获取当前屏幕截图
    im = ImageGrab.grab()
    # 转为opencv的BGR格式
    npim = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    # 获取屏幕中人脸所在区域
    face_locations = face_recognition.face_locations(npim)
    # 提取面部特征
    face_encodings = face_recognition.face_encodings(npim, face_locations)
    # 储存截取到的人物姓名列表，匹配不上就是Unknown
    names = []

    # 遍历获取到的特征和数据库中的做匹配
    for face_encoding in face_encodings:
        matchs = face_recognition.compare_faces(user_faces_encodings, face_encoding)
        name = "Unknown"
        for index, is_match in enumerate(matchs):
            if is_match:
                name = user_names[index]
                break
        names.append(name)

    # 遍历人脸所在区域，并画框标识姓名
    for (top, right, bottom, left), name in zip(face_locations, names):
        # 设置框的颜色，格式B, G, R
        color = (0,255,0)
        # 如果是目标就改变框的颜色
        if name in targer:
            color = (0, 0, 255)
        # 在人像区域画框，格式cv2.rectangle(图像, (left, top), (right, bottom), 颜色, 线的粗细)
        cv2.rectangle(npim, (left, top), (right, bottom), color, 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        # 标识
        cv2.putText(npim, name, (left, top-10), font, 0.5, color, 1)
    # 将标识的画面显示出来
    cv2.imshow('Video', npim)
    # 如果按q就退出while循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 退出程序时释放所有窗口
cv2.destroyAllWindows()
