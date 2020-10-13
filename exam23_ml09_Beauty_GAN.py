#!/usr/bin/env python
# coding: utf-8




import numpy as np
import dlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#dlib 설치 방법
#anconda prompt에서 activate '환경이름' 으로 가서 
#conda install -c conda-forge dlib 를 입력해 준다





detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('../models/shape_predictor_5_face_landmarks.dat') #모양을 예측해주는 모델





img = dlib.load_rgb_image('../imgs/12.jpg')
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()





#사진의 얼굴부분만 찾는법
img_result = img.copy()  #copy를 해주는 이유는 원본은 보존하기 위해서
dets = detector(img) #얼굴영역을 찾아준다. 
if len(dets) == 0:
    print('cannot find face')
else:
    fig, ax = plt.subplots(1, figsize=(16,10))
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x,y),w,h,linewidth=2, edgecolor='r', facecolor='none') 
        ax.add_patch(rect)
    ax.imshow(img_result)    
    plt.show()





fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections() #얼굴이 돌려져 있거나 하는 사진을 똑바로 쳐다보는 사진으로 돌려주는 코드
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r')
        ax.add_patch(circle)
        
ax.imshow(img_result)





#얼굴 사진만 뽑아서 정렬 시키는 법
faces = dlib.get_face_chips(img, objs, size=256, padding=0.3) #padding이 들어가는 이유는 이미지 사이의 간격을 위해서
fig, axes = plt.subplots(1,len(faces)+1, figsize = (20,16)) #len은 사진에 들어있는 얼굴의 개수를 나타내는데 +1을 해주는 이유는 원본사진을 앞에 두기 위함 이다
axes[0].imshow(img)

for i, face in enumerate(faces):
    axes[i+1].imshow(face)




#이미지를 주면 이미지의 있는 얼굴을 찾어서 쭉 정렬해 주는 함수를 만드는 코드(위에 한것들 합친것)
def align_faces(img):
    dets = detector(img, 1) #여기에서 이미지의 얼굴을 찾아서 img에서 넣어주었다. 
    objs = dlib.full_object_detections() #사진의 위치정보를 찾는 코드(여기에서는 사진속에 얼굴의 위치정보)
    
    for detection in dets:
        s = sp(img, detection) #이미지에서 landmark(점)를 가져온다
        objs.append(s) #가져온 landmark를 얼굴의 위치정보에 붙여준다. 
    faces = dlib.get_face_chips(img, objs, size=256, padding = 0.35) #이미지에서 얼굴 잘라서 보여주는 코드
    return faces

test_img = dlib.load_rgb_image('../imgs/12.jpg')
test_faces = align_faces(test_img)
fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20,16))
axes[0].imshow(test_img)

for i, face in enumerate(test_faces):
    axes[i+1].imshow(face)





#얼굴 이미지에다가 화장을 해주는 코드
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #모델 초기화 해주는 코드

saver = tf.train.import_meta_graph('../models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('../models'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')




def preprocess(img):
    return(img / 255. - 0.5) * 2

def deprocess(img):
    return(img + 1) / 2




#화장 안한 이미지 (source image)
img1 = dlib.load_rgb_image('../imgs/no_makeup/xfsy_0405.png')
img1_faces = align_faces(img1)

#화장한 이미지 (reference image)
img2 = dlib.load_rgb_image('../imgs/makeup/002.jpg')
img2_faces = align_faces(img2)

fig, axes = plt.subplots(1,2,figsize=(16,10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()




#얼굴 이미지에다가 화장을 해주는 코드
src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0)

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img})
output_img = deprocess(output[0])

fig, axes  = plt.subplots(1,3, figsize=(20,10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)

plt.show()
