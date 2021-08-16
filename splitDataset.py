import os
import shutil

# path in cui hai le foto delle x del dataset da dividere
pathOriginali = "C:\\Users\\guerr\\PycharmProjects\\TesiSegmentazionePinna\\dataset\\originaleSquared\\input\\"
# path in cui hai le foto delle y del dataset da dividere
pathContorni = "C:\\Users\\guerr\\PycharmProjects\\TesiSegmentazionePinna\\dataset\\originaleSquared\\output\\"

# path in cui devi spostare le foto per validazione e training
pathVal = "C:\\Users\\guerr\\PycharmProjects\\TesiSegmentazionePinna\\dataset\\originaleSquared\\val"
pathTrain = "C:\\Users\\guerr\\PycharmProjects\\TesiSegmentazionePinna\\dataset\\originaleSquared\\train"
input = "\\input\\"
output = "\\output\\"


os.chdir(pathOriginali)
lista = os.listdir()
i = 0
validation = 20


for elem in lista:
    filename = elem[: len(elem) -4 ]
    sourceFile = pathOriginali + elem
    sourceEdge = pathContorni + filename + ".png"
    if i % validation == 0:
        shutil.move(sourceFile, pathVal + input + elem)
        shutil.move(sourceEdge, pathVal + output + filename + ".png")
    else:
        shutil.move(sourceFile, pathTrain + input + elem)
        shutil.move(sourceEdge, pathTrain + output + filename + ".png")

    i += 1