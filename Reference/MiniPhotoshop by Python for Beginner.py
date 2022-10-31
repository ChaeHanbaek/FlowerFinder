import tkinter
import tkinter.filedialog
import tkinter.simpledialog
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# 함수 선언
# 이미지 출력


def displayImage(img, Iwidth, Iheight):  # 이미지 주소, 너비, 높이
    global window, canvas, paper, photo, photo2, oriX, oriY

    window.geometry(str(Iwidth)+"x"+str(Iheight))  # 여는 이미지에 맞게 창크기 조절
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
    canvas.pack()

# 파일 열기


def func_open():
    global window, canvas, paper, photo, photo2, oriX, oriY
    readFp = tkinter.filedialog.askopenfilename(parent=window, filetypes=(
        ("모든 그림 파일", ".jpg; *.jpeg; *.bmp; *.png; *.tif; *.gif"), ("모든파일", "*.*")))
    photo = Image.open(readFp).convert("RGB")
    oriX = photo.width
    oriY = photo.height
    photo2 = photo.copy()  # 원본을 복사해서 이미지 처리
    newX = photo2.width
    newY = photo2.height
    displayImage(photo2, newX, newY)

# 파일 저장


def func_save():
    global window, canvas, paper, photo, photo2, oriX, oriY

    if photo2 == None:  # 파일이 없으면 되돌림
        return
    saveFp = tkinter.filedialog.asksaveasfile(parent=window, mode='w', defaultextension=".jpg", filetypes=(
        ("JPG 파일", "*.jpg;*.jpeg"), ("모든 파일", "*.*")))

    photo2.save(saveFp.name)

# 종료


def func_exit():
    global window
    window.destroy()

# 이미지 확대


def func_zoomin():
    global window, canvas, paper, photo, photo2, oriX, oriY
    scale = tkinter.simpledialog.askinteger(
        "확대 배율", "배율(2~4)을 입력하세요.", minvalue=2, maxvalue=4)
    photo2 = photo.copy()
    photo2 = photo2.resize((int(oriX*scale), int(oriY*scale)))
    newX = photo2.width
    newY = photo2.height
    displayImage(photo2, newX, newY)

# 이미지 축소
def func_zoomout():
    global window, canvas, paper, photo, photo2, oriX, oriY
    scale = tkinter.simpledialog.askinteger(
        "축소 배율", "배율(2~4)을 입력하세요.", minvalue=2, maxvalue=4)
    photo2 = photo.copy()
    photo2 = photo2.resize((int(oriX/scale), int(oriY/scale)))
    newX = photo2.width
    newY = photo2.height
    displayImage(photo2, newX, newY)

# 이미지 상하 반전
def func_mirror1():
    global window, canvas, paper, photo, photo2, oriX, oriY
    photo2 = photo.copy()
    photo2 = photo2.transpose(Image.FLIP_TOP_BOTTOM)
    newX = photo2.width
    newY = photo2.height
    displayImage(photo2, newX, newY)

# 이미지 좌우 반전
def func_mirror2():
    global window, canvas, paper, photo, photo2, oriX, oriY
    photo2 = photo.copy()
    photo2 = photo2.transpose(Image.FLIP_LEFT_RIGHT)
    newX = photo2.width
    newY = photo2.height
    displayImage(photo2, newX, newY)

# 이미지 회전
def func_rotate():
    global window, canvas, paper, photo, photo2, oriX, oriY
    degree = tkinter.simpledialog.askinteger(
        "회전 각도", "각도(0~360)을 입력하세요.", minvalue=0, maxvalue=360)
    photo2 = photo.copy()
    photo2 = photo2.rotate(degree, expand=True)
    newX = photo2.width
    newY = photo2.height
    displayImage(photo2, newX, newY)

# 이미지 밝게
def func_bright():
    global window, canvas, paper, photo, photo2, oriX, oriY
    value = tkinter.simpledialog.askfloat(
        "밝게", "값(1.0~10.0)을 입력하세요.", minvalue=1.0, maxvalue=10.0)
    photo2 = photo.copy()
    photo2 = ImageEnhance.Brightness(photo2).enhance(value)
    newX = photo2.width
    newY = photo2.height
    displayImage(photo2, newX, newY)

# 이미지 어둡게
def func_dark():
    global window, canvas, paper, photo, photo2, oriX, oriY
    value = tkinter.simpledialog.askfloat(
        "어둡게", "값(0~1.0)을 입력하세요.", minvalue=0, maxvalue=1.0)
    photo2 = photo.copy()
    photo2 = ImageEnhance.Brightness(photo2).enhance(value)
    newX = photo2.width
    newY = photo2.height
    displayImage(photo2, newX, newY)

# 이미지 블러링
def func_blur():
    global window, canvas, paper, photo, photo2, oriX, oriY
    photo2 = photo.copy()
    photo2 = photo2.filter(ImageFilter.BLUR)
    newX = photo2.width
    newY = photo2.height
    displayImage(photo2, newX, newY)

# 이미지 엠보싱
def func_embo():
    global window, canvas, paper, photo, photo2, oriX, oriY
    photo2 = photo.copy()
    photo2 = photo2.filter(ImageFilter.EMBOSS)
    newX = photo2.width
    newY = photo2.height
    displayImage(photo2, newX, newY)

# 이미지 흑백
def func_bw():
    global window, canvas, paper, photo, photo2, oriX, oriY
    photo2 = photo.copy()
    photo2 = ImageOps.grayscale(photo2)
    newX = photo2.width
    newY = photo2.height
    displayImage(photo2, newX, newY)


# 전역변수 설정
window, canvas, paper = None, None, None  # 메인 윈도우, 캔버스, 출력전
photo, photo2 = None, None  # 원본사진, 처리후 사진
oriX, oriY = None, None  # 원본 사진 폭과 높이

# 메인 윈도우
window = tkinter.Tk()
window.geometry("250x250")  # 메인 윈도우 해상도 설정
window.title("Mini-Photoshop-like")

# 메뉴 구성
mainMenu = tkinter.Menu(window)
window.config(menu=mainMenu)

# 파일메뉴
fileMenu = tkinter.Menu(mainMenu)
mainMenu.add_cascade(label="파일", menu=fileMenu)
fileMenu.add_command(label="파일 열기", command=func_open)
fileMenu.add_command(label="파일 저장", command=func_save)
fileMenu.add_separator()  # 메뉴에 줄 추가
fileMenu.add_command(label="종료", command=func_exit)

# 이미지메뉴 1
image1Menu = tkinter.Menu(mainMenu)
mainMenu.add_cascade(label="이미지 처리(1)", menu=image1Menu)
image1Menu.add_command(label="확대", command=func_zoomin)
image1Menu.add_command(label="축소", command=func_zoomout)
image1Menu.add_separator()
image1Menu.add_command(label="상하 반전", command=func_mirror1)
image1Menu.add_command(label="좌우 반전", command=func_mirror2)
image1Menu.add_command(label="회전", command=func_rotate)

# 이미지메뉴 2
image2Menu = tkinter.Menu(mainMenu)
mainMenu.add_cascade(label="이미지 처리(2)", menu=image2Menu)
image2Menu.add_command(label="밝게", command=func_bright)
image2Menu.add_command(label="어둡게", command=func_dark)
image2Menu.add_separator()
image2Menu.add_command(label="블러링", command=func_blur)
image2Menu.add_command(label="엠보싱", command=func_embo)
image2Menu.add_separator()
image2Menu.add_command(label="흑백", command=func_bw)

window.mainloop()
