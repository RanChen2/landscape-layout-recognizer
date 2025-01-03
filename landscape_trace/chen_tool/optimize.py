import json
import cv2
import numpy as np
import math
from PIL import Image
from scipy.interpolate import splprep, splev
class optimize:

    def __init__(self, image):
        self.image = image
        self.result = list()

    def bezier_curve(self,p0, p1, p2, p3, inserted):
        """
        三阶贝塞尔曲线
        p0, p1, p2, p3 - 点坐标，tuple、list或numpy.ndarray类型
        inserted  - p0和p3之间插值的数量
        """
        assert isinstance(p0, (tuple, list, np.ndarray))
        assert isinstance(p0, (tuple, list, np.ndarray))
        assert isinstance(p0, (tuple, list, np.ndarray))
        assert isinstance(p0, (tuple, list, np.ndarray))

        if isinstance(p0, (tuple, list)):
            p0 = np.array(p0)
        if isinstance(p1, (tuple, list)):
            p1 = np.array(p1)
        if isinstance(p2, (tuple, list)):
            p2 = np.array(p2)
        if isinstance(p3, (tuple, list)):
            p3 = np.array(p3)

        points = list()
        for t in np.linspace(0, 1, inserted + 2):
            points.append(p0 * np.power((1 - t), 3) + 3 * p1 * t * np.power((1 - t), 2) + 3 * p2 * (1 - t) * np.power(t,
                                                                                                                      2) + p3 * np.power(
                t, 3))

        return np.vstack(points)

    def contour_sparse(self, contour, min_dist=3):
        def two_point_dist(point1, point2):
            return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        sparse_contour = list()
        pre_idx = 0
        post_idx = 1
        sparse_contour.append(contour[pre_idx])
        while post_idx < len(contour):
            dist = two_point_dist(contour[pre_idx][0], contour[post_idx][0])
            if dist >= min_dist:
                sparse_contour.append(contour[post_idx])
                pre_idx = post_idx
                post_idx += 1
            else:
                post_idx += 1

        return np.array(sparse_contour)

    def smoothing_base_bezier(self,date_x, date_y, k=0.5, inserted=10, closed=False):
        """
         基于三阶贝塞尔曲线的数据平滑算法
         date_x  - x维度数据集，list或numpy.ndarray类型
         date_y  - y维度数据集，list或numpy.ndarray类型
         k   - 调整平滑曲线形状的因子，取值一般在0.2~0.6之间。默认值为0.5
         inserted - 两个原始数据点之间插值的数量。默认值为10
         closed  - 曲线是否封闭，如是，则首尾相连。默认曲线不封闭
         """

        assert isinstance(date_x, (list, np.ndarray))
        assert isinstance(date_y, (list, np.ndarray))

        if isinstance(date_x, list) and isinstance(date_y, list):
            assert len(date_x) == len(date_y), u'x数据集和y数据集长度不匹配'
            date_x = np.array(date_x)
            date_y = np.array(date_y)
        elif isinstance(date_x, np.ndarray) and isinstance(date_y, np.ndarray):
            assert date_x.shape == date_y.shape, u'x数据集和y数据集长度不匹配'
        else:
            raise Exception(u'x数据集或y数据集类型错误')

        # 第1步：生成原始数据折线中点集
        mid_points = list()
        for i in range(1, date_x.shape[0]):
            mid_points.append({
                'start': (date_x[i - 1], date_y[i - 1]),
                'end': (date_x[i], date_y[i]),
                'mid': ((date_x[i] + date_x[i - 1]) / 2.0, (date_y[i] + date_y[i - 1]) / 2.0)
            })

        if closed:
            mid_points.append({
                'start': (date_x[-1], date_y[-1]),
                'end': (date_x[0], date_y[0]),
                'mid': ((date_x[0] + date_x[-1]) / 2.0, (date_y[0] + date_y[-1]) / 2.0)
            })

        # 第2步：找出中点连线及其分割点
        split_points = list()
        for i in range(len(mid_points)):
            if i < (len(mid_points) - 1):
                j = i + 1
            elif closed:
                j = 0
            else:
                continue

            x00, y00 = mid_points[i]['start']
            x01, y01 = mid_points[i]['end']
            x10, y10 = mid_points[j]['start']
            x11, y11 = mid_points[j]['end']
            d0 = np.sqrt(np.power((x00 - x01), 2) + np.power((y00 - y01), 2))
            d1 = np.sqrt(np.power((x10 - x11), 2) + np.power((y10 - y11), 2))
            k_split = 1.0 * d0 / (d0 + d1)

            mx0, my0 = mid_points[i]['mid']
            mx1, my1 = mid_points[j]['mid']

            split_points.append({
                'start': (mx0, my0),
                'end': (mx1, my1),
                'split': (mx0 + (mx1 - mx0) * k_split, my0 + (my1 - my0) * k_split)
            })

        # 第3步：平移中点连线，调整端点，生成控制点
        crt_points = list()
        for i in range(len(split_points)):
            vx, vy = mid_points[i]['end']  # 当前顶点的坐标
            dx = vx - split_points[i]['split'][0]  # 平移线段x偏移量
            dy = vy - split_points[i]['split'][1]  # 平移线段y偏移量

            sx, sy = split_points[i]['start'][0] + dx, split_points[i]['start'][1] + dy  # 平移后线段起点坐标
            ex, ey = split_points[i]['end'][0] + dx, split_points[i]['end'][1] + dy  # 平移后线段终点坐标

            cp0 = sx + (vx - sx) * k, sy + (vy - sy) * k  # 控制点坐标
            cp1 = ex + (vx - ex) * k, ey + (vy - ey) * k  # 控制点坐标

            if crt_points:
                crt_points[-1].insert(2, cp0)
            else:
                crt_points.append([mid_points[0]['start'], cp0, mid_points[0]['end']])

            if closed:
                if i < (len(mid_points) - 1):
                    crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end']])
                else:
                    crt_points[0].insert(1, cp1)
            else:
                if i < (len(mid_points) - 2):
                    crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end']])
                else:
                    crt_points.append(
                        [mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end'], mid_points[i + 1]['end']])
                    crt_points[0].insert(1, mid_points[0]['start'])

        # 第4步：应用贝塞尔曲线方程插值
        out = list()
        for item in crt_points:
            group = self.bezier_curve(item[0], item[1], item[2], item[3], inserted)
            out.append(group[:-1])

        out.append(group[-1:])
        out = np.vstack(out)

        return out.T[0], out.T[1]

    def bezier(self, contour, k, inserted):

        x_list = list()
        y_list = list()
        for point in contour:
            x_list.append(point[0][0])
            y_list.append(point[0][1])

        x_numpy = np.array(x_list)
        y_numpy = np.array(y_list)
        x_curve, y_curve = self.smoothing_base_bezier(x_numpy, y_numpy, k=k, inserted=inserted, closed=True)

        curve_list = list()
        for x, y in zip(x_curve, y_curve):
            curve_list.append([[int(x), int(y)]])
        contour = np.array(curve_list)

        return contour

    def convert_to_cv(self,image):
        if isinstance(image, Image.Image):
            # 如果输入是PIL图像，则转换为NumPy数组
            image_np = np.array(image)
            # 将RGB转换为BGR格式，因为OpenCV使用BGR格式
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            return image_cv
        elif isinstance(image, np.ndarray):
            # 如果输入已经是NumPy数组，则直接返回
            return image


    def run(self):
        # 读取图像文件



        image = self.convert_to_cv(self.image)
        # 将图像转换为灰度图
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 对灰度图进行二值化处理
        _, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

        # 创建一个3x3的全1结构元素
        kernel = np.ones(shape=[3, 3], dtype=np.uint8)
        # 对二值图像进行膨胀操作，迭代两次
        image_binary = cv2.dilate(image_binary, kernel, iterations=2)
        # 对膨胀后的图像进行腐蚀操作，恢复图像形态
        image_binary = cv2.erode(image_binary, kernel=kernel)

        # 寻找轮廓
        contours, _ = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 遍历找到的轮廓
        for contour in contours:
            # 如果轮廓面积小于或等于20，忽略此轮廓
            if cv2.contourArea(contour) <= 20:
                continue
            # 对轮廓点进行多边形逼近
            contour = cv2.approxPolyDP(contour, 3, True)
            # 对轮廓进行贝塞尔曲线处理
            contour = self.bezier(contour, k=0.2, inserted=1)  # 默认值是k=0.4，inserted=100 比较密集
            # 对轮廓进行稀疏处理，减少点的数量
            contour = self.contour_sparse(contour, min_dist=100)
            # 再次检查轮廓面积，如果处理后面积仍小于或等于20，忽略此轮廓
            if cv2.contourArea(contour) <= 20:
                continue

            # # 在原图上绘制处理后的轮廓
            # cv2.drawContours(image, contour, -1, (0, 0, 255,), 20)
            # # 将轮廓点添加到结果列表中
            # print('con: ',[point[0] for point in contour.tolist()])
            self.result.append([point[0] for point in contour.tolist()])

        if len(self.result) > 1:            #只取最长的边
            # 找出最长的元素
            longest = max(self.result, key=len)
            # 更新result为只包含最长的那个元素
            self.result = [longest]


        # # 将绘制了轮廓的图像保存到文件
        # cv2.imwrite(self.contours_path, image)
        # # 将结果数据写入JSON文件
        # with open(self.result_path, 'w', encoding='utf-8') as js:
        #     json.dump(self.result, js, indent=4, ensure_ascii=False)


def cubic_spline_interpolation(points, num=10):
    # print(len(points))
    # print(points)

    if len(points) < 4:
        return []
    else:
        points = points

        # print(len(points))
        # print(points)
        x = np.array([point[0] for point in points])
        y = np.array([point[1] for point in points])
        # 使用三次样条插值
        tck, u = splprep([x, y], k=3, s=0)
        u_new = np.linspace(0, 1, num=len(points) * num)
        x_spline, y_spline = splev(u_new, tck)

        return [(int(round(x)), int(round(y))) for x, y in zip(x_spline, y_spline)]