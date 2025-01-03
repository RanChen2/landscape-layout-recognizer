import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import splprep, splev
from chen_tool.bezier import bezier_1 as bezier
from math import sqrt

def resize_image(img, max_size=1024):
    """调整图像大小，保持比例"""
    width, height = img.size
    if width > height:
        ratio = max_size / width
    else:
        ratio = max_size / height
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return img.resize((new_width, new_height), Image.BICUBIC)

def cv2_preprocess(image):
    """OpenCV图像预处理"""
    image_np = np.array(image)
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones(shape=[3, 3], dtype=np.uint8)
    image_binary = cv2.dilate(image_binary, kernel, iterations=2)
    image_binary = cv2.erode(image_binary, kernel=kernel)
    return image_binary

def cubic_spline_interpolation(points, num=10):
    """三次样条插值"""
    if len(points) < 4:
        return []
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    tck, u = splprep([x, y], k=3, s=0)
    u_new = np.linspace(0, 1, num=len(points) * num)
    x_spline, y_spline = splev(u_new, tck)
    return [(int(round(x)), int(round(y))) for x, y in zip(x_spline, y_spline)]

def total_length(points_list):
    """计算点列表的总长度"""
    return sum(sqrt((points_list[i][0] - points_list[i-1][0]) ** 2 + 
               (points_list[i][1] - points_list[i-1][1]) ** 2) 
               for i in range(1, len(points_list)))

def optimize(image, min_dist, k, inserted, closed, ismask):
    """优化轮廓点"""
    result = []
    image_binary = cv2_preprocess(image)
    contours, _ = cv2.findContours(image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) <= 20:
            continue

        contour = cv2.approxPolyDP(contour, 3, closed)
        contour = bezier(contour=contour, k=k, inserted=inserted, closed=closed)

        sparse_contour = []
        pre_idx = 0
        post_idx = 1
        sparse_contour.append(contour[pre_idx])
        
        while post_idx < len(contour):
            dist = sqrt((contour[pre_idx][0][0] - contour[post_idx][0][0]) ** 2 +
                       (contour[pre_idx][0][1] - contour[post_idx][0][1]) ** 2)
            if dist >= min_dist:
                sparse_contour.append(contour[post_idx])
                pre_idx = post_idx
                post_idx += 1
            else:
                post_idx += 1

        contour = np.array(sparse_contour)
        if cv2.contourArea(contour) <= 20:
            continue

        result.append([point[0] for point in contour.tolist()])
    
    if ismask and len(result) > 1:
        longest = max(result, key=len)
        result = [longest]
    
    return result

def pre_process(name, mask, ismask, input_image_size):
    """预处理图像"""
    if name == 'mask':
        min_dist, k, inserted = 100, 0.2, 1
    else:
        min_dist, k, inserted = 1, 0.4, 100
        
    if name == 'ZW':
        mask = cv2_preprocess(mask)
        mask = Image.fromarray(mask)
    else:
        points_data = optimize(mask, min_dist=min_dist, k=k, inserted=inserted, closed=True, ismask=ismask)
        canvas = Image.new('RGB', input_image_size, color='black')
        draw = ImageDraw.Draw(canvas)
        
        if name == 'PZ+DL':
            points_data = sorted(points_data, key=total_length, reverse=True)
            seen = []
            for index, points_list in enumerate(points_data):
                if points_list not in seen:
                    seen.append(points_list)
                    if len(points_list) >= 4:
                        fill = 'white' if index == 0 else 'black'
                        smoothed_points = cubic_spline_interpolation(points_list)
                        draw.polygon(smoothed_points, fill=fill, outline=None)
        else:
            for points_list in points_data:
                if len(points_list) >= 4:
                    smoothed_points = cubic_spline_interpolation(points_list)
                    draw.polygon(smoothed_points, fill='white', outline='white')
        
        mask = canvas.resize(input_image_size)
    
    return mask 