import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

list_images = os.listdir('source_img')


def return_points(gray, cornerness, rows, cols):
    th_top_left, th_top_right = -1e6, -1e6
    th_bottom_left, th_bottom_right = -1e6, -1e6

    opt_top_left, opt_top_right = None, None
    opt_bottom_left, opt_bottom_right = None, None

    quad_size = 7

    for r in range(quad_size, rows-quad_size):
        for c in range(quad_size, cols-quad_size):
            if cornerness[r, c] < -7:
                continue
            
            block = 255*gray[r-quad_size:r+quad_size+1, c-quad_size:c+quad_size+1]
            
            quad_top_left = block[0:quad_size, 0:quad_size]
            quad_top_right = block[0:quad_size, quad_size+1:2*quad_size+1]
            quad_bottom_left = block[quad_size+1:2*quad_size+1, 0:quad_size]
            quad_bottom_right = block[quad_size+1:2*quad_size+1, quad_size+1:2*quad_size+1]
            
            descriptor = np.mean(quad_bottom_right) - np.mean(quad_top_left) - np.mean(quad_top_right) - np.mean(quad_bottom_left)
            if descriptor > th_top_left:
                th_top_left = descriptor
                opt_top_left = (c, r)
            
            descriptor = np.mean(quad_bottom_left) - np.mean(quad_top_left) - np.mean(quad_top_right) - np.mean(quad_bottom_right)
            if descriptor > th_top_right:
                th_top_right = descriptor
                opt_top_right = (c, r)
            
            descriptor = np.mean(quad_top_right) - np.mean(quad_top_left) - np.mean(quad_bottom_left) - np.mean(quad_bottom_right)
            if descriptor > th_bottom_left:
                th_bottom_left = descriptor
                opt_bottom_left = (c, r)
            
            descriptor = np.mean(quad_top_left) - np.mean(quad_top_right) - np.mean(quad_bottom_left) - np.mean(quad_bottom_right)
            if descriptor > th_bottom_right:
                th_bottom_right = descriptor
                opt_bottom_right = (c, r)
    return opt_top_left, opt_top_right, opt_bottom_left, opt_bottom_right


def transform_image(img, opt_top_left, opt_top_right, opt_bottom_left, opt_bottom_right):
    pt_A = opt_top_left
    pt_B = opt_bottom_left
    pt_C = opt_bottom_right
    pt_D = opt_top_right

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
    
    
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    dst = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])
    M = cv2.getPerspectiveTransform(np.float32([pt_A, pt_B, pt_C, pt_D]), dst)

    rectified = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return rectified


def image_preload(img_path):
    img = cv2.imread(img_path)
    img_origin = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    scale_percent = 30
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)/255
    rows, cols = gray.shape

    cornerness = cv2.cornerHarris(gray, 2, 3, 0.04)
    cornerness = np.maximum(cornerness, 0)
    cornerness = np.log(cornerness + 1e-7)

    opt_top_left, opt_top_right, opt_bottom_left, opt_bottom_right = return_points(gray, cornerness, rows, cols)

    scale_percent_upscale = 100 / scale_percent
    origin_opt_top_left = (int(opt_top_left[0] * scale_percent_upscale), int(opt_top_left[1] * scale_percent_upscale))
    origin_opt_top_right = (int(opt_top_right[0] * scale_percent_upscale), int(opt_top_right[1] * scale_percent_upscale))
    origin_opt_bottom_left = (int(opt_bottom_left[0] * scale_percent_upscale), int(opt_bottom_left[1] * scale_percent_upscale))
    origin_opt_bottom_right = (int(opt_bottom_right[0] * scale_percent_upscale), int(opt_bottom_right[1] * scale_percent_upscale))

    # print(origin_opt_top_left, origin_opt_top_right, origin_opt_bottom_left, origin_opt_bottom_right)

    # out_origin = img_origin.copy()
    # out_origin = cv2.circle(out_origin, origin_opt_top_left, 30, (255,0,0), -1)
    # out_origin = cv2.circle(out_origin, origin_opt_top_right, 30, (255,0,0), -1)
    # out_origin = cv2.circle(out_origin, origin_opt_bottom_left, 30, (255,0,0), -1)
    # out_origin = cv2.circle(out_origin, origin_opt_bottom_right, 30, (255,0,0), -1)
    # plt.imshow(out_origin)
    # plt.show()

    t_img = transform_image(img_origin, origin_opt_top_left, origin_opt_top_right, origin_opt_bottom_left, origin_opt_bottom_right)

    cv2.imwrite('output_img/' + img_path.split('/')[-1], cv2.cvtColor(t_img, cv2.COLOR_RGB2BGR))
    # plt.imshow(t_img)
    # plt.show()

    


for img_name in list_images:
    print("Processing image: " + img_name)
    image_preload("source_img/" + img_name)
    print("Image " + img_name + " processed successfully")

print("All images processed successfully")

# image_preload("source_img/IMG_0779.jpeg")