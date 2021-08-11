import cv2
import os
import numpy as np
import PIL.Image
from PIL import ImageEnhance


# per ogni immagine presente nella cartella crea una foto più luminosa e una meno luminosa
def imageBrightener(pathImmagine, pathContorno, pathSalvataggio, pathSalvataggioContorno):
    os.chdir(pathImmagine)
    files = os.listdir()
    chiara = 1.25
    scura = 0.75
    i = 1
    lenFiles = len(files)

    for file in files:
        print(f'Immagine {i} di {lenFiles}')
        img = PIL.Image.open(pathImmagine + "\\" + file)
        # image brightness enhancer
        enhancer = ImageEnhance.Brightness(img)
        im_output = enhancer.enhance(scura)
        if im_output.mode != 'RGB':
            im_output = im_output.convert('RGB')

        save = f'{pathSalvataggio}\\{file[:len(file) - 4]}_darkened.jpg'
        opencvImage = cv2.cvtColor(np.array(im_output), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save, opencvImage)
        contorno = cv2.imread(f'{pathContorno}\\{file[:len(file) - 4]}.png')
        cv2.imwrite(f'{pathSalvataggioContorno}\\{file[:len(file) - 4]}_darkened.png', contorno)
        im_output2 = enhancer.enhance(chiara)
        opencvImage2 = cv2.cvtColor(np.array(im_output2), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{pathSalvataggio}\\{file[:len(file) - 4]}_brightened.jpg', opencvImage2)
        cv2.imwrite(f'{pathSalvataggioContorno}\\{file[:len(file) - 4]}_brightened.png', contorno)
        i += 1


# per ogni immagine presente nella cartella crea una foto più luminosa e una meno luminosa
def imageContrast(pathImmagine, pathContorno, pathSalvataggio, pathSalvataggioContorno):
    os.chdir(pathImmagine)
    files = os.listdir()
    chiara = 1.25
    scura = 0.75
    i = 1
    lenFiles = len(files)

    for file in files:
        print(f'Immagine {i} di {lenFiles}')
        img = PIL.Image.open(pathImmagine + "\\" + file)
        # image brightness enhancer
        enhancer = ImageEnhance.Contrast(img)
        im_output = enhancer.enhance(scura)
        if im_output.mode != 'RGB':
            im_output = im_output.convert('RGB')

        save = f'{pathSalvataggio}\\{file[:len(file) - 4]}_lessContrast.jpg'
        opencvImage = cv2.cvtColor(np.array(im_output), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save, opencvImage)
        contorno = cv2.imread(f'{pathContorno}\\{file[:len(file) - 4]}.png')
        cv2.imwrite(f'{pathSalvataggioContorno}\\{file[:len(file) - 4]}_lessContrast.png', contorno)
        im_output2 = enhancer.enhance(chiara)
        opencvImage2 = cv2.cvtColor(np.array(im_output2), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{pathSalvataggio}\\{file[:len(file) - 4]}_moreContrast.jpg', opencvImage2)
        cv2.imwrite(f'{pathSalvataggioContorno}\\{file[:len(file) - 4]}_moreContrast.png', contorno)
        i += 1


# rupta l'immagine di un angolo dato in input
def rotateAngle(img, angle, color):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = img.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_img = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_img[0,0])
    abs_sin = abs(rotation_img[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_img[0, 2] += bound_w/2 - image_center[0]
    rotation_img[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation imgrix
    rotated_img = cv2.warpAffine(img, rotation_img, (bound_w, bound_h), borderValue=color)
    return rotated_img



# crea tutte le rotazioni dell'immagine di partenza
def createImageRotations(path, pathSalvataggio, color, extension):
    angles = [30, 45, 60, 120, 150, 270]
    os.chdir(path)
    files = os.listdir()
    i = 1
    for file in files:
        print("Immagine numero: " + str(i) + "su 515")
        filePath = path + "\\" + file
        savePath = pathSalvataggio + "\\" + file
        print(savePath)
        original = cv2.imread(filePath)
        if original is None:
            stream = open(filePath, "rb")
            bytesArray = bytearray(stream.read())
            numpyarray = np.asarray(bytesArray, dtype=np.uint8)
            original = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        for angle in angles:
            img = rotateAngle(original, angle, color)
            cv2.imwrite(savePath[:len(savePath) - 4] + "_" + str(angle) + extension, img)
        i = i + 1


# permette di specchiare le immagini
def flipImages(path, pathSalvataggio, extension):
    os.chdir(path)
    files = os.listdir()
    i = 1
    for file in files:
        print("Immagine numero: " + str(i))
        filePath = path + file
        savePath = pathSalvataggio + "\\" + file
        print(savePath)
        original = cv2.imread(filePath)
        if original is None:
            stream = open(filePath, "rb")
            bytesArray = bytearray(stream.read())
            numpyarray = np.asarray(bytesArray, dtype=np.uint8)
            original = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        img = cv2.flip(original, 1)
        cv2.imwrite(savePath[:len(savePath) - 4] + "_flipped" + extension, img)
        i = i + 1


# salvare immagini e aprirle con cv2
# per ogni immagine
# per ogni angolo
# ruota immagine e salva

if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    pathContorni = os.path.join(dirname, 'Dataset\\Contorni\\')
    pathNuoviContorni =os.path.join(dirname, 'Dataset\\ContorniRotazione\\')
    pathOriginali = os.path.join(dirname, 'Dataset\\JPEGImages\\')
    pathOriginaliRotazione = os.path.join(dirname, 'Dataset\\JPEGRotazione\\')



    createImageRotations(pathContorni, pathNuoviContorni, (0,0,0), '.png')
    createImageRotations(pathOriginali, pathOriginaliRotazione, (0,0,255), '.jpg')

    print("Nuovi contorni")
    flipImages(pathNuoviContorni, pathNuoviContorni, ".png")
    print("Contorni")
    flipImages(pathContorni, pathNuoviContorni, ".png")
    print("Ruotate")
    flipImages(pathOriginaliRotazione, pathOriginaliRotazione, ".jpg")
    print("Originali")
    flipImages(pathOriginali, pathOriginaliRotazione, ".jpg")



    imageBrightener(pathOriginaliRotazione, pathNuoviContorni, pathOriginaliRotazione, pathNuoviContorni)
    imageBrightener(pathOriginali, pathContorni, pathOriginaliRotazione, pathNuoviContorni)
    imageContrast(pathOriginaliRotazione, pathNuoviContorni, pathOriginaliRotazione, pathNuoviContorni)
    imageContrast(pathOriginali, pathContorni, pathOriginaliRotazione, pathNuoviContorni)


