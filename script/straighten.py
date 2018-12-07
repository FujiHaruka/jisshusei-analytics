import cv2
import numpy as np

def straighten(img):
    """
    聴取票画像の大枠だけを取り出してまっすぐ長方形に伸ばした画像にする

    Paramaters
    ----------
    img: ndarray (2 dim)

    Returns
    -------
    img: ndarray (2 dim)
    """
    # ノイズ除去
    sigma_x = 5
    average_square = (sigma_x, sigma_x)
    img_gauss = cv2.GaussianBlur(img, average_square, sigma_x)
    ret, thres = cv2.threshold(img_gauss, 200, 255, cv2.THRESH_BINARY_INV)

    # 輪郭を抽出
    _, contours, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭のうち面積最大のもの（＝大枠）を取得
    max_area_cnt = max(contours, key=lambda cnt: cv2.contourArea(cnt))

    # 大枠の角4点を取得
    points = np.float32([
        min(max_area_cnt, key=lambda position: position[0][0] + position[0][1]),
        max(max_area_cnt, key=lambda position: position[0][0] - position[0][1]),
        max(max_area_cnt, key=lambda position: position[0][0] + position[0][1]),
        min(max_area_cnt, key=lambda position: position[0][0] - position[0][1]),
    ])

    # 射影変換で大枠を画像の4隅に合わせる
    rows,cols = img.shape
    points2 = np.float32([[10, 10], [cols-10, 10], [cols-10, rows-10], [10, rows-10]])
    M = cv2.getPerspectiveTransform(points,points2)

    dst = cv2.warpPerspective(img,M,(cols, rows))

    return dst
