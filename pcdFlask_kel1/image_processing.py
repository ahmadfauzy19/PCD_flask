import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from skimage.morphology import skeletonize
from collections import Counter
from pylab import savefig
import cv2
import os


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


def zoomout():
    img = loadImage040("static/img/img_now.jpg")
    persen = 70
    img = cvResizeImage040(img,persen)
    img_arr = np.asarray(img)
    img = Image.fromarray(img_arr)
    img.save("static/img/img_now.jpg")


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
    out = np.zeros((h_img-2, w_img-2), dtype=np.float32)  # Menggunakan np.float32 untuk array keluaran
    new_img = np.zeros((h_img-2, w_img-2, 3), dtype=np.uint8)
    
    for channel in range(3):
        array = img[:, :, channel]
        for h in range(h_img-2):
            for w in range(w_img-2):
                S = np.multiply(array[h:h+3, w:w+3], kernel)
                out[h, w] = np.sum(S)
        out_ = np.clip(out, 0, 255)
        new_img[:, :, channel] = out_
    
    return new_img


def edge_detection():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.uint8)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now_edge_detected.jpg")


def blur():
    img = cv2.imread("static/img/img_now.jpg")
    blurred_img = cv2.GaussianBlur(img, (15, 15), 0)  # Adjust the kernel size as needed
    cv2.imwrite("static/img/img_now_blurred.jpg", blurred_img)



def sharpening():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.uint8)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now_sharpened.jpg")


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
    img = loadImage040("static/img/img_now.jpg")
    kernel = np.ones((3,3), np.uint8)
    dilated_image = cv2.dilate(img, kernel, iterations=1)
    new_img = Image.fromarray(dilated_image)
    new_img.save("static/img/img_now.jpg")

def erosi():
    img = loadImage040("static/img/img_now.jpg")
    kernel = np.ones((3,3), np.uint8)
    dilated_image = cv2.erode(img, kernel, iterations=1)
    new_img = Image.fromarray(dilated_image)
    new_img.save("static/img/img_now.jpg")

def opening():

    img = loadImage040("static/img/img_now.jpg")
    kernel = np.ones((3,3), np.uint8)
    opening_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    new_img = Image.fromarray(opening_image)
    new_img.save("static/img/img_now.jpg")

def closing():
    img = loadImage040("static/img/img_now.jpg")
    kernel = np.ones((3,3), np.uint8)
    opening_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    new_img = Image.fromarray(opening_image)
    new_img.save("static/img/img_now.jpg")




def counting():
    img = loadImage040("static/img/img_now.jpg")
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    erosi_img = cv2.erode(threshold_img, kernel, iterations=3)
    contours, _ = cv2.findContours(erosi_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gerigi_with_contours = img.copy()
    cv2.drawContours(gerigi_with_contours, contours, -1, (0, 255, 0), 2)
    num_blobs = len(contours)
    new_img = Image.fromarray(gerigi_with_contours)
    new_img.save("static/img/img_now.jpg")
    return num_blobs

def ekstra_citra(lokasi_citra):
    citra = cv2.imread(lokasi_citra, cv2.IMREAD_GRAYSCALE)
    _, citra_biner = cv2.threshold(citra, 128, 255, cv2.THRESH_BINARY_INV)
    kontur, _ = cv2.findContours(citra_biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    kontur_terbesar = max(kontur, key=cv2.contourArea)
    return kontur_terbesar

def penipisan_citra(lokasi_citra):
    citra = cv2.imread(lokasi_citra, cv2.IMREAD_GRAYSCALE)
    _, citra_biner = cv2.threshold(citra, 128, 255, cv2.THRESH_BINARY_INV)
    citra_penipisan = skeletonize(citra_biner)
    citra_penipisan = citra_penipisan.astype(np.uint8)
    kontur, _ = cv2.findContours(citra_penipisan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    citra_penipisan = max(kontur, key=cv2.contourArea)
    return citra_penipisan

def hitung_freeman_chain_code(kontur):
    kode_chain = []
    titik_valid = []
    arah = [0, 7, 6, 5, 4, 3, 2, 1]
    perubahan = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    # Mendapatkan titik-titik kontur
    titik = [tuple(poin[0]) for poin in kontur]

    # Menemukan titik awal (titik paling kiri)
    indeks_awal = titik.index(min(titik, key=lambda x: x[0]))

    titik_sekarang = titik[indeks_awal]
    titik_awal = titik[indeks_awal]

    # Mengikuti kontur dan menghasilkan kode chain
    while True:
        found = False
        ra={0,1,2,3,4,5,6,7}
        for i in ra:
            next_point = (titik_sekarang[0] + perubahan[i][0], titik_sekarang[1] + perubahan[i][1])
            if next_point in titik:
                found = True
                if next_point in titik_valid:
                    continue
                else :
                    titik_valid.append(next_point)
                    break
        if not found:
            break
        kode_chain.append(arah[i])  
        titik_sekarang = next_point
        if titik_sekarang == titik_awal:
            break

    return kode_chain


def simpan_ke_json(basis_pengetahuan, nama_file):
    with open(nama_file, "w") as file:
        json.dump(basis_pengetahuan, file)

def simpan_gambar():
    kode_chain_freeman_digit = {}
    kode_chain_penipisan_digit = {}
    nama_citra = []
    nama_citra = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]
    for digit, nama in enumerate(nama_citra):
        lokasi_citra = f"static/assets/images/daftar_gambar/{nama}.png"
        kontur = ekstra_citra(lokasi_citra)
        kode_chain = hitung_freeman_chain_code(kontur)
        
        kode_chain_freeman_digit[digit] = kode_chain
        citra_penipisan = penipisan_citra(lokasi_citra)
        kode_chain = hitung_freeman_chain_code(citra_penipisan)
        kode_chain_penipisan_digit[digit] = kode_chain

    simpan_ke_json(kode_chain_freeman_digit, "knowledge_base_metode1.json")
    simpan_ke_json(kode_chain_penipisan_digit, "knowledge_base_metode2.json")

    return kode_chain_freeman_digit, kode_chain_penipisan_digit

def kenali_digit(kode_chain, basis_pengetahuan):
    jarak_minimum = float('inf')
    digit_terkenali = None
    for digit, referensi_kode_chain in basis_pengetahuan.items():
        jarak = sum(1 for a, b in zip(kode_chain, referensi_kode_chain) if a != b)
        if jarak < jarak_minimum:
            jarak_minimum = jarak
            digit_terkenali = digit
    return digit_terkenali

def uji_pengenalan_angka(daftar_citra_uji, basis_pengetahuan, metode):
    digit=[]
    for lokasi_citra_uji in daftar_citra_uji:
        if metode == "penipisan":
            kode_chain_uji = hitung_freeman_chain_code(penipisan_citra(lokasi_citra_uji))
        else: 
            kode_chain_uji = hitung_freeman_chain_code(ekstra_citra(lokasi_citra_uji))
        digit_terkenali = kenali_digit(kode_chain_uji, basis_pengetahuan)
        digit.append(digit_terkenali)
    joined_digits = ''.join(map(str, digit))
    return joined_digits

def deteksi_gambar(daftar_gambar,metode):
    ekstra , penipisan = simpan_gambar()
    daftar_citra_uji = [('static/img/deteksi/'+ gambar.filename) for gambar in daftar_gambar]
    print("daftar gambar bos =", daftar_citra_uji)
    if metode == "penipisan":
        knowladge = penipisan
    else:
        knowladge = ekstra

    hasil = uji_pengenalan_angka(daftar_citra_uji, knowladge, metode)
    return hasil

