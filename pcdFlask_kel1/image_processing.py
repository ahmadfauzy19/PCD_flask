import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from collections import Counter
from pylab import savefig
import cv2


#modul modul pembantu
def loadImage040(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def cvResizeImage040(img, scale_percent):
    # Mendapatkan dimensi gambar
    height, width, channels = img.shape
    
    # Menghitung persentase dari lebar dan tinggi yang baru
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    
    # Mengubah ukuran gambar sesuai dengan persentase
    resized_img = cv2.resize(img, (new_width, new_height))
    
    return resized_img


def grayscale():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]
    new_arr = r.astype(int) + g.astype(int) + b.astype(int)
    new_arr = (new_arr/3).astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    for i in range(w):
        for j in range(h):
            r, g, b = im.getpixel((i, j))
            if r != g != b:
                return False
    return True


def zoomin():
    img = loadImage040("static/img/img_now.jpg")
    persen = 130
    img = cvResizeImage040(img,persen)
    img_arr = np.asarray(img)
    img = Image.fromarray(img_arr)
    img.save("static/img/img_now.jpg")
    # img = Image.open("static/img/img_now.jpg")
    # img = img.convert("RGB")
    # img_arr = np.asarray(img)
    # new_size = ((img_arr.shape[0] * 2),
    #             (img_arr.shape[1] * 2), img_arr.shape[2])
    # new_arr = np.full(new_size, 255)
    # new_arr.setflags(write=1)

    # r = img_arr[:, :, 0]
    # g = img_arr[:, :, 1]
    # b = img_arr[:, :, 2]

    # new_r = []
    # new_g = []
    # new_b = []

    # for row in range(len(r)):
    #     temp_r = []
    #     temp_g = []
    #     temp_b = []
    #     for i in r[row]:
    #         temp_r.extend([i, i])
    #     for j in g[row]:
    #         temp_g.extend([j, j])
    #     for k in b[row]:
    #         temp_b.extend([k, k])
    #     for _ in (0, 1):
    #         new_r.append(temp_r)
    #         new_g.append(temp_g)
    #         new_b.append(temp_b)

    # for i in range(len(new_arr)):
    #     for j in range(len(new_arr[i])):
    #         new_arr[i, j, 0] = new_r[i][j]
    #         new_arr[i, j, 1] = new_g[i][j]
    #         new_arr[i, j, 2] = new_b[i][j]

    # new_arr = new_arr.astype('uint8')
    # new_img = Image.fromarray(new_arr)
    # new_img.save("static/img/img_now.jpg")


def zoomout():
    img = loadImage040("static/img/img_now.jpg")
    persen = 70
    img = cvResizeImage040(img,persen)
    img_arr = np.asarray(img)
    img = Image.fromarray(img_arr)
    img.save("static/img/img_now.jpg")
    # img = Image.open("static/img/img_now.jpg")
    # img = img.convert("RGB")
    # x, y = img.size
    # new_arr = Image.new("RGB", (int(x / 2), int(y / 2)))
    # r = [0, 0, 0, 0]
    # g = [0, 0, 0, 0]
    # b = [0, 0, 0, 0]

    # for i in range(0, int(x/2)):
    #     for j in range(0, int(y/2)):
    #         r[0], g[0], b[0] = img.getpixel((2 * i, 2 * j))
    #         r[1], g[1], b[1] = img.getpixel((2 * i + 1, 2 * j))
    #         r[2], g[2], b[2] = img.getpixel((2 * i, 2 * j + 1))
    #         r[3], g[3], b[3] = img.getpixel((2 * i + 1, 2 * j + 1))
    #         new_arr.putpixel((int(i), int(j)), (int((r[0] + r[1] + r[2] + r[3]) / 4), int(
    #             (g[0] + g[1] + g[2] + g[3]) / 4), int((b[0] + b[1] + b[2] + b[3]) / 4)))
    # new_arr = np.uint8(new_arr)
    # new_img = Image.fromarray(new_arr)
    # new_img.save("static/img/img_now.jpg")


def move_left():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
    g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
    b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_right():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
    g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
    b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_up():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
    g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
    b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_down():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    b = np.pad(b, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_addition():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr+100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_substraction():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr-100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_multiplication():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr*1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_division():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr/1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def convolution(img, kernel):
    h_img, w_img, _ = img.shape
    out = np.zeros((h_img-2, w_img-2), dtype=np.float)
    new_img = np.zeros((h_img-2, w_img-2, 3))
    if np.array_equal((img[:, :, 1], img[:, :, 0]), img[:, :, 2]) == True:
        array = img[:, :, 0]
        for h in range(h_img-2):
            for w in range(w_img-2):
                S = np.multiply(array[h:h+3, w:w+3], kernel)
                out[h, w] = np.sum(S)
        out_ = np.clip(out, 0, 255)
        for channel in range(3):
            new_img[:, :, channel] = out_
    else:
        for channel in range(3):
            array = img[:, :, channel]
            for h in range(h_img-2):
                for w in range(w_img-2):
                    S = np.multiply(array[h:h+3, w:w+3], kernel)
                    out[h, w] = np.sum(S)
            out_ = np.clip(out, 0, 255)
            new_img[:, :, channel] = out_
    new_img = np.uint8(new_img)
    return new_img


def edge_detection():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def blur():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int)
    kernel = np.array(
        [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def sharpening():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def hitung_histogram(img_path):
    img = loadImage040(img_path)
    # Bagi saluran warna
    if is_grey_scale(img_path):
        hist_gray = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.figure() 
        plt.bar(range(256), hist_gray.ravel(), color='gray')  # Histogram untuk skala abu-abu
        plt.savefig(f'static/img/grey_histogram.jpg', dpi=300)

    r, g, b = cv2.split(img)

    # Hitung histogram dari setiap saluran warna
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    data_rgb = [hist_r, hist_g, hist_b]
    warna = ['red', 'green', 'blue']
    
    for i, data in enumerate(data_rgb):
        plt.figure()  # Membuat gambar histogram baru
        plt.bar(range(256), data.ravel(), color=warna[i])  # Histogram untuk setiap saluran warna
        plt.savefig(f'static/img/{warna[i]}_histogram.jpg', dpi=300)
        plt.clf()  


def histogram_rgb():
    img_path = "static/img/img_now.jpg"
    hitung_histogram(img_path)


def df(img):  # to make a histogram (count distribution frequency)
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i, j]] += 1
    return values


def cdf(hist):  # cumulative distribution frequency
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1]+hist[i]
    # Now we normalize the histogram
    # What your function h was doing before
    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf


def histogram_equalizer():
    img = cv2.imread('static\img\img_now.jpg', 0)
    my_cdf = cdf(df(img))
    # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
    image_equalized = np.interp(img, range(0, 256), my_cdf)
    cv2.imwrite('static/img/img_now.jpg', image_equalized)


def threshold(lower_thres, upper_thres):
    img = loadImage040("static/img/img_now.jpg")
    image =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Menggunakan metode thresholding
    _, binary_image = cv2.threshold(image, lower_thres,upper_thres , cv2.THRESH_BINARY)
    new_img = Image.fromarray(binary_image)
    new_img.save("static/img/img_now.jpg")

def dilasi():
    return 1

def erosi():
    return 1




