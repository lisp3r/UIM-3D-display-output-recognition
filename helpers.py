
import numpy
import cv2

from collections import defaultdict


class GeneralRecognitionFailure(Exception):
    """ Main exception type for this tool """

def show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def show_contours(image, contours, ids):
    copy_to_show = image.copy()
    if type(ids) == int:
        ids = [ids]
    for id in ids:
        cv2.drawContours(copy_to_show, contours, id, (255, 0, 0), 2, cv2.LINE_AA)
    show('Contours {}'.format(ids), copy_to_show)

def draw_symbols(image, symbols):
    height, width, _ = image.shape
    for i in range(0,len(symbols)):
        x = int((i+0.5)*width/len(symbols))
        y = int(height/10)
        cv2.putText(image, str(symbols[i]), (x,y), cv2.FONT_HERSHEY_DUPLEX, int(height/200), (255, 0, 0), int(height/200))

def prepare_edged(image, blur_filter_size=7, canny_th1=0, canny_th2=100):
    """
    Make edged image from original one
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_filter_size, blur_filter_size), 0)
    edge = cv2.Canny(blur, canny_th1, canny_th2, 255)
    return edge

def get_polygon_ids(contours, corner_cnt=4, epsilon=0.04):
    """
    Return IDs of contours who has corner_cnt corners
    :param corner_cnt: number of countours required
    :param espilon: share of perimeter for approximation
    """
    ids = []
    for i in range(0,len(contours)):
        peri = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon*peri, True)
        if len(approx) == corner_cnt and cv2.isContourConvex(approx):
            ids.append(i)
    return ids

def get_square_ids(contours, epsilon=0.1):
    """
    Return ids of contours with 4 angles and somewhat width == length
    :param contours: array of contours
    :param epsilon: how much different in width and length is ok
    """
    rects_ids = get_polygon_ids(contours, corner_cnt=4)
    square_ids = []
    for rc_id in rects_ids:
        rect = contours[rc_id]
        x, y = get_size(rect)
        if abs(x-y) < epsilon*x:
            square_ids.append(rc_id)
    return square_ids

def get_size(contour, epsilon=0.02):
    """
    Get contours (x, y) size
    :param epsilon: approximation parameter
    return: (x,y) size
    """
    # Make somewhat approximation to shrink points count
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon*peri, True)
    # Get axis coord values
    x_vals = [ p[0][0] for p in approx]
    y_vals = [ p[0][1] for p in approx]
    # Return min-max
    return max(x_vals) - min(x_vals), max(y_vals) - min(y_vals)

def get_area(contour, epsilon=0.02):
    """
    Get rectangle contour area based on size (see below)
    """
    x,y = get_size(contour, epsilon=epsilon)
    return x*y

def get_immediate_children(id, bad_hierarchy):
    """
    Return ids of the first childlrens for a contour with ID id
    :param id: id of contour we finding childrens for
    :param bad_hierarchy: hierarchy of contour
    """
    hierarchy = bad_hierarchy[0]
    ids = []
    first_ch = hierarchy[id][2]
    if first_ch:
        ch_id = first_ch
    else:
        return []
    while ch_id > 0:
        ids.append(ch_id)
        ch_id = hierarchy[ch_id][0]
    ch_id = first_ch
    while ch_id > 0:
        if ch_id != first_ch:
            ids.append(ch_id)
        ch_id = hierarchy[ch_id][1]
    return ids

def get_all_children(id, hierarchy):
    ids = get_immediate_children(id, hierarchy)
    for ch_id in ids:
        subchildren = get_all_children(ch_id, hierarchy)
        if subchildren:
            ids.extend(subchildren)
    return ids

def get_mass_center(contour):
    ms = cv2.moments(contour)
    # add 1e-5 to avoid division by zero
    return (ms['m10'] / (ms['m00'] + 1e-5), ms['m01'] / (ms['m00'] + 1e-5))

def sorted_by_axis(ctrs):
    """
    Sort contours by X axis
    """
    mc_ctrs = [{'mc': get_mass_center(c)[0], 'ctr': c} for c in ctrs]
    mc_ctrs.sort(key=lambda x: x['mc'])
    return [mc_ctr['ctr'] for mc_ctr in mc_ctrs]

def strip_contour(image, contour, border=0.05):
    """ Strip square contour from image, cut border %'s from each side """
    true_rect = cv2.minAreaRect(contour)
    box = numpy.int0(cv2.boxPoints(true_rect))

    width, height = get_size(box)
    x_border = int(width * border)
    y_border = int(height * border)

    x1 = min([p[0] for p in box]) + x_border
    y1 = min([p[1] for p in box]) + y_border
    x2 = max([p[0] for p in box]) - x_border
    y2 = max([p[1] for p in box]) - y_border

    return image[y1:y2, x1:x2]
