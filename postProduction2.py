import matplotlib.pyplot as plt
import os
from skimage import morphology
import numpy as np
import skimage
from PIL import Image

COSTANTE_CARRASSI = 255
INTORNO = 50

stock_dir = os.path.join('result', 'output_filtered')
save_path = os.path.join('result', 'postProduction')
stock_files = os.listdir(stock_dir)

if not os.path.exists(save_path):
    os.makedirs(save_path)

for file in stock_files:
    image = plt.imread(os.path.join(stock_dir, file))
    tmp = skimage.color.rgb2gray(image)
    tmp = np.where(tmp > 0.1, 1, 0)

    tmp = morphology.binary_closing(tmp, morphology.square(5))
    tmp = morphology.remove_small_objects(tmp.astype(bool), min_size=50, connectivity=2).astype(int)

    last_start = -1
    last_end = -1

    for row in range(tmp.shape[0]):
        first = -1
        last = -1
        right = tmp.shape[1] - 1
        left = 0

        left_border_found = False
        right_border_found = False

        while (left_border_found is False or right_border_found is False) and left <= right:

            if left_border_found is False:
                if tmp[row][left] != 0:
                    if last_start == -1 or (last_start - INTORNO <= left <= last_start + INTORNO):
                        first = left
                else:
                    if first != -1:
                        left_border_found = True

                left += 1

            if right_border_found is False:
                if tmp[row][right] != 0:
                    if last_end == -1 or (last_end - INTORNO <= right <= last_end + INTORNO):
                        last = right
                else:
                    if last != -1:
                        right_border_found = True
                right -= 1

        if right_border_found is True and left_border_found is True:
            last_end = last
            last_start = first

        if first != -1 and last != -1:
            for i in range(first + 5, last - 5):
                tmp[row][i] = 0

    tmp = morphology.binary_closing(tmp, morphology.square(20))
    tmp = morphology.remove_small_objects(tmp.astype(bool), min_size=COSTANTE_CARRASSI, connectivity=2).astype(int)

    mask_x, mask_y = np.where(tmp == 0)
    image[mask_x, mask_y, :3] = 0

    final_img = Image.fromarray(np.uint8(image * 255))
    final_img.save(os.path.join(save_path, file))

    # plot the result

    plt.figure(figsize=(10, 10))

    plt.subplot(221), plt.imshow(plt.imread(os.path.join(stock_dir, file))), plt.title(file, size=20), plt.axis('off')
    plt.subplot(222), plt.imshow(image), plt.title('output', size=20), plt.axis('off')

    plt.show()
