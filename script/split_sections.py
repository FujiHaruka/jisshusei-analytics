import cv2
import numpy as np

def split_sections(img):
    """
    聴取票画像の大枠からセクションを分割する

    Paramaters
    ----------
    img: ndarray (2 dim)

    Returns
    -------
    section_imgs: ndarray (2 dim) tuple (7 len)
    """
    # ノイズ除去
    sigma_x = 5
    average_square = (sigma_x, sigma_x)
    img_gauss = cv2.GaussianBlur(img, average_square, sigma_x)
    ret, thres = cv2.threshold(img_gauss, 200, 255, cv2.THRESH_BINARY_INV)

    # 輪郭を抽出
    _, contours, _ = cv2.findContours(thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # セクションの領域を選ぶ
    max_area_cnt = max(contours, key=lambda cnt: cv2.contourArea(cnt))
    max_area = cv2.contourArea(max_area_cnt)
    # 最大面積の領域と一定以上小さな領域を除外する
    def filter_area(cnt):
        area = cv2.contourArea(cnt)
        return 200000 < area and area < cv2.contourArea(max_area_cnt)
    section_contours = list(filter(filter_area, contours))
    if len(section_contours) != 7:
        raise Exception("Section is 7, but {}".format(len(section_contours)))

    # セクションを長方形に近似
    section_rects = [ cv2.boundingRect(cnt) for cnt in section_contours ]
    section_rects = sorted(section_rects, key=lambda rect: rect[1]) # y でソート

    # セクションの画像を切り取る
    def trim_img(rect):
        x,y,w,h = rect
        return img[y:y+h,x:x+w]
    section_imgs = [ trim_img(rect) for rect in section_rects ]
    return section_imgs

if __name__ == "__main__":
    img = cv2.imread("src.jpeg", 0)
    imgs = split_sections(img)
    for i, dst in enumerate(imgs):
        cv2.imwrite("out/out_{}.jpeg".format(i), dst)
