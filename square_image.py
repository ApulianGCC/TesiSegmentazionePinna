import os
import PIL.Image


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = PIL.Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = PIL.Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def createSquareImages(path, save_path, background_color):
    for filename in os.listdir(path):
        img = PIL.Image.open(path + "\\" + filename)
        if img is not None:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            square_img = expand2square(img, background_color)
            square_img.save(save_path + filename)
    return


if __name__ == '__main__':
    print("Eseguito script come main")

    img = PIL.Image.open('result/input/finValidation.jpg')
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    square_img = expand2square(img, (255, 0, 0))
    square_img.save('finValidationSquared.jpg')

    print('stop')

    dirname = os.path.dirname(__file__)
    pathInput = os.path.join(dirname, 'dataset\\originale\\input\\')

    pathSaveInput = os.path.join(dirname, 'dataset\\originaleSquared\\input\\')

    if not os.path.exists(pathSaveInput):
        os.makedirs(pathSaveInput)

    createSquareImages(pathInput, pathSaveInput, (255, 0, 0))

    pathOutput = os.path.join(dirname, 'dataset\\originale\\output\\')
    pathSaveOutput = os.path.join(dirname, 'dataset\\originaleSquared\\output\\')

    if not os.path.exists(pathSaveOutput):
        os.makedirs(pathSaveOutput)

    createSquareImages(pathOutput, pathSaveOutput, (0, 0, 0))
