import matplotlib.pyplot as plt
import os
from skimage import morphology
import numpy as np
import skimage
from PIL import Image

COSTANTE_CARRASSI = 255
'''
img = plt.imread('difficile.png')
grayscale = skimage.color.rgb2gray(img)
binarized = np.where(grayscale > 0.1 , 1, 0)


tmp = morphology.remove_small_objects(binarized.astype(bool), min_size=100, connectivity=2).astype(int)
# black out pixels
tmp_mask_x, tmp_mask_y = np.where(tmp == 0)
binarized[tmp_mask_x,tmp_mask_y] = 0

binarized = morphology.binary_closing(binarized, morphology.square(16))

plt.figure()
plt.imshow(binarized, cmap='gray')
plt.show()

processed = morphology.remove_small_objects(binarized.astype(bool), min_size=500, connectivity=2).astype(int)
# black out pixels
mask_x, mask_y = np.where(processed == 0)
img[mask_x, mask_y, :3] = 0



# plot the result
plt.figure(figsize=(10, 10))
plt.subplot(131), plt.imshow(plt.imread('difficile.png')), plt.title('input', size=20), plt.axis('off')
plt.subplot(133), plt.imshow(img), plt.title('output', size=20), plt.axis('off')
plt.show()



image = plt.imread('difficile.png')
tmp = skimage.color.rgb2gray(image)
tmp = np.where(tmp > 0.1 , 1, 0)

tmp = morphology.binary_closing(tmp, morphology.square(5))
tmp = morphology.remove_small_objects(tmp.astype(bool), min_size=50, connectivity=2).astype(int)


for row in range(tmp.shape[0]):
    first = -1
    last = -1
    end = tmp.shape[1] - 1
    start = 0

    x = False
    y = False

    while (x == False or y == False) and start <= end:
        if x is False:
            if tmp[row][start] != 0:
                first = start
            else:
                if first != -1:
                    x = True

            start +=1

        if y is False:
            if tmp[row][end] != 0:
                last = end
            else:
                if last != -1:
                    y = True
            end -=1
    if  first != -1 and last != -1:
        for i in range (first+5, last-5):
            tmp[row][i] = 0

tmp = morphology.binary_closing(tmp, morphology.square(20))
tmp = morphology.remove_small_objects(tmp.astype(bool), min_size=150, connectivity=2).astype(int)

mask_x, mask_y = np.where(tmp == 0)
image[mask_x, mask_y, :3] = 0

plt.figure()
plt.imshow(image, cmap = 'gray')
plt.title('nicholas')
plt.show()


'''
stock_dir = os.path.join('result', 'output')
save_path = os.path.join('result', 'postProduction')
stock_files = os.listdir(stock_dir)
for file in stock_files:
    image = plt.imread(os.path.join(stock_dir, file))
    tmp = skimage.color.rgb2gray(image)
    tmp = np.where(tmp > 0.1, 1, 0)

    tmp = morphology.binary_closing(tmp, morphology.square(5))
    tmp = morphology.remove_small_objects(tmp.astype(bool), min_size=50, connectivity=2).astype(int)

    for row in range(tmp.shape[0]):
        first = -1
        last = -1
        end = tmp.shape[1] - 1
        start = 0

        x = False
        y = False

        while (x == False or y == False) and start <= end:
            if x is False:
                if tmp[row][start] != 0:
                    first = start
                else:
                    if first != -1:
                        x = True

                start += 1

            if y is False:
                if tmp[row][end] != 0:
                    last = end
                else:
                    if last != -1:
                        y = True
                end -= 1
        if first != -1 and last != -1:
            for i in range(first + 5, last - 5):
                tmp[row][i] = 0

    tmp = morphology.binary_closing(tmp, morphology.square(20))
    tmp = morphology.remove_small_objects(tmp.astype(bool), min_size=COSTANTE_CARRASSI, connectivity=2).astype(int)

    mask_x, mask_y = np.where(tmp == 0)
    image[mask_x, mask_y, :3] = 0

    final_img = Image.fromarray(np.uint8(image*255))
    final_img.save(os.path.join(save_path, file))

    # plot the result

    plt.figure(figsize=(10, 10))

    plt.subplot(221), plt.imshow( plt.imread(os.path.join(stock_dir, file))), plt.title(file, size=20), plt.axis('off')
    plt.subplot(222), plt.imshow(image), plt.title('output', size=20), plt.axis('off')

    plt.show()

