# tensorflow, tf.keras 임포트
from msilib.schema import AdvtExecuteSequence
from tensorflow import keras
from tensorflow.python.keras.models import load_model

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                              
from PIL import Image

#GUI
import tkinter
import tkinter.filedialog
import tkinter.simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#path
import sys
import os
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

# 함수 선언
# 이미지 출력
def displayImage(img):  # 이미지 주소, 너비, 높이
    global window, canvas, paper, photo, photo2, oriX, oriY
    Iwidth = img.width
    Iheight = img.height

    if canvas != None:  # 이미 캔버스가 있으면 삭제
        canvas.destroy()

    canvas = tkinter.Canvas(window, width=Iwidth, height=Iheight)
    paper = tkinter.PhotoImage(width=Iwidth, height=Iheight)
    canvas.create_image((Iwidth/2, Iheight/2), image=paper, state="normal")
    rgbString = ""
    rgbImage = img.convert('RGB')
    for y in range(0, Iheight):
        tmpString = ""
        for x in range(0, Iwidth):
            r, g, b = rgbImage.getpixel((x, y))
            tmpString += "#%02x%02x%02x " % (r, g, b)  # x뒤 한칸 공백
        rgbString += "{" + tmpString + "} "  # }뒤 한칸 공백
    paper.put(rgbString)
    canvas.place(x=10,y=10)

# 파일 열기
def func_open():
    global window, canvas, paper, photo
    readFp = tkinter.filedialog.askopenfilename(parent=window, filetypes=(
        ("모든 그림 파일", ".jpg; *.jpeg; *.bmp; *.png; *.tif; *.gif"), ("모든파일", "*.*")))
    photo = Image.open(readFp).convert("RGB")
    cvimg = PIL2OpenCV(photo)
    photo=photo.resize((400,300))
    displayImage(photo)

    pd = cvt_flower_recognition(cvimg)
    plot_image(pd,Flower_Label)
    plot_value_array(pd)


# 종료
def func_exit():
    global window
    window.destroy()


#PIL과 openCv2 상호 전환 함수, PIL로 안되는 이미지 작업은 OpenCV2로 처리해야 한다.
def PIL2OpenCV(pil_image):
    # open image using PIL

    # use numpy to convert the pil_image into a numpy array
    numpy_image=np.array(pil_image)  

    # convert to a openCV2 image and convert from RGB to BGR format
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    #display image to GUI
    #cv2.imshow("PIL2OpenCV",opencv_image)

    return opencv_image

def OpenCV2PIL(opencv_image):

    #display image to GUI
    #cv2.imshow("PIL2OpenCV", opencv_image)

    # convert from BGR to RGB
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # convert from openCV2 to PIL
    pil_image=Image.fromarray(color_coverted)

    return pil_image

#이미지 파일을 변환시켜 예측형태로 가공
def cvt_flower_recognition(_img):
    global model
    r_img = cv2.cvtColor(_img,cv2.COLOR_BGR2RGB)
    r_img = cv2.resize(r_img, (64,64))
    r_img = r_img[:,:,:]
    r_img = r_img.reshape((1, 64, 64, 3))
    r_img = r_img.astype('float32') / 255
    prediction = model.predict(r_img)
    return prediction

# 각 클래스에 대한 예측을 표시하기 위한 함수
def plot_image(predictions_array,label_List):

    predictions_array= predictions_array[0]

    predicted_index = np.argmax(predictions_array)
    second_index = second_max_index(predictions_array)
    third_index = third_max_index(predictions_array)

    first_label.config(text="1st_Prediction: {}, {:2.0f}%".format(label_List[predicted_index],
                                100*np.max(predictions_array)))
    second_label.config(text="2nd_Preidction: {}, {:2.0f}%".format(label_List[second_index],
                                predictions_array[second_index]*100))
    third_label.config(text="3rd_Preidction: {}, {:2.0f}%".format(label_List[third_index],
                                predictions_array[third_index]*100))                                

#예측을 그래프 형태로 보여줌
def plot_value_array(predictions_array):
    global Flower_Label
    predictions_array=predictions_array[0]
    label = range(16)

    figure1 = plt.Figure(figsize=(4, 3), dpi=100)
    bar1 = FigureCanvasTkAgg(figure1, window)
    axes = figure1.add_subplot()
    axes.bar(label, predictions_array, color="#777777")
    axes.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    axes.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    axes.set_title("Probability")

    thisplot = axes.bar(label, predictions_array, color="#777777")
    predicted_index = np.argmax(predictions_array)
    second_index = second_max_index(predictions_array)
    third_index = third_max_index(predictions_array)
    thisplot[predicted_index].set_color('red')
    thisplot[second_index].set_color('blue')
    thisplot[third_index].set_color('green')

    axes.text(predicted_index, predictions_array[predicted_index], Flower_Label[predicted_index], horizontalalignment='center')
    axes.text(second_index, predictions_array[second_index], Flower_Label[second_index], horizontalalignment='center')
    axes.text(third_index, predictions_array[third_index], Flower_Label[third_index], horizontalalignment='center')
    bar1.get_tk_widget().place(x=420,y=10)

def second_max_index(_list):
    s_list = _list.tolist()
    s_list.sort(reverse=True)
    secondMax= s_list[1]
    for i in range(len(_list)):
        if secondMax == _list[i]:
            return i

def third_max_index(_list):
    s_list = _list.tolist()
    s_list.sort(reverse=True)
    secondMax= s_list[2]
    for i in range(len(_list)):
        if secondMax == _list[i]:
            return i            

# 전역변수 설정
window, canvas, paper = None, None, None  # 메인 윈도우, 캔버스, 출력전
photo= None, None  # 원본사진
Flower_Label=['Astilbe','Bellflower','Black_eyed_susan','Calendula','California_poppy','Carnation','Common_daisy','Coreopsis','Daffodil','Dandelion','Iris','Magnolia','Rose','Sunflower','Tulip','Water_lily']

# 메인 윈도우
window = tkinter.Tk()
window.geometry("950x380")  # 메인 윈도우 해상도 설정
window.title("FlowerFinder")

# 메뉴 구성
mainMenu = tkinter.Menu(window)
window.config(menu=mainMenu)

# 아이콘 로딩
icon_FileOpen = tkinter.PhotoImage(file="{}\\icon\\FileOpen.png".format(dir_path))
icon_Exit = tkinter.PhotoImage(file="{}\\icon\\Exit.png".format(dir_path))

# 파일메뉴
fileMenu = tkinter.Menu(mainMenu)
mainMenu.add_cascade(label="파일", menu=fileMenu)
fileMenu.add_command(label="파일 열기", image=icon_FileOpen, compound='left', command=func_open)
fileMenu.add_separator()  # 메뉴에 줄 추가
fileMenu.add_command(label="종료", image = icon_Exit, compound='left', command=func_exit)

#캔버스,라벨 설정
canvas = tkinter.Canvas(window,width=400,height=300,background='white')
first_label = tkinter.Label(window,text='1st')
second_label = tkinter.Label(window,text='2nd')
third_label = tkinter.Label(window,text='3rd')

figure1 = plt.Figure(figsize=(4, 3), dpi=100)  
bar1 = FigureCanvasTkAgg(figure1, window)
axes = figure1.add_subplot()
axes.set_title("Probability")

button_FileOpen = tkinter.Button(window, image=icon_FileOpen, command=func_open)
button_Exit = tkinter.Button(window, image=icon_Exit, command=func_exit)

flower_text = "Flower_Label\n"
for i in range(len(Flower_Label)):
    flower_text += "{}: {}\n".format(i,Flower_Label[i])
label_Label=tkinter.Label(window,text=flower_text,justify='left')

canvas.place(x=10,y=10)
first_label.place(x=10,y=310)
second_label.place(x=10,y=330)
third_label.place(x=10,y=350)
bar1.get_tk_widget().place(x=420,y=10)
button_FileOpen.place(x=830, y=10)
button_Exit.place(x=900, y=10)
label_Label.place(x=830,y=50)

#FlowerFinder 모델 로딩
#용량을 줄이기위해 모델만든후 가중치만 로딩
IMG_SIZE=64
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (IMG_SIZE,IMG_SIZE,3))) #img_size바꿨으면 여기서도 수정해줘야함
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same',activation ='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(filters = 256, kernel_size = (5,5),padding = 'Same',activation ='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(filters = 512, kernel_size = (5,5),padding = 'Same',activation ='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512))   
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(16, activation = "softmax")) 
model.load_weights("{}\\FF_Model.h5".format(dir_path))

window.mainloop()