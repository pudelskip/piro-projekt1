import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import find_contours, approximate_polygon
import math



def normHeights(points,nr):
    points = np.array(points)
    maxH = np.max(points)
    minH = np.min(points)
    if maxH-minH==0:
        return points

    points = map(lambda p : (p-minH)*200/(maxH-minH) , points)
    return np.array(list(points))


def rotate_bound(image, angle):

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))

def unstreatch(image):
    (rows, cols) = image.shape[:2]
    yToCheck = rows//4
    xCheck = cols//2

    i = 1
    left = np.min(np.argwhere(image[-i, :] > 0))
    right = np.max(np.argwhere(image[-i, :] > 0))

    while left > cols//4:
        i+=1
        left = np.min(np.argwhere(image[-i, :] > 0))
    j=1
    while right < 3*cols//4:
        j += 1
        right = np.max(np.argwhere(image[-j, :] > 0))


    pts1 = np.float32([[0, 0], [left+1, rows - i], [right-1, rows - j], [cols - 1, 0]])
    pts2 = np.float32([[0, 0], [0, rows - i], [cols - 1, rows - j], [cols - 1, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst1 = cv2.warpPerspective(image, M, (cols, rows))


    if (rows>20):
        upperLeft = np.min(np.argwhere(dst1[-yToCheck, :] > 200))
        upperRight = np.max(np.argwhere(dst1[-yToCheck, :] > 200))


        if upperLeft > 1 and upperLeft<xCheck:
            m = yToCheck / upperLeft
            newX = int(rows/m)
            pts1 = np.float32([[newX,0],[0,rows-1],[cols-1,rows-1],[cols-1,0]])
            pts2 = np.float32([[0,0],[0,rows-1],[cols-1,rows-1],[cols-1,0]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst1 = cv2.warpPerspective(dst1, M, (cols, rows))


        if upperRight<cols-1 and upperRight>xCheck:
            m = yToCheck / (upperRight-cols)
            newX = int(rows / m)+(cols)
            pts1 = np.float32([[0, 0], [0, rows - 1], [cols - 1, rows - 1], [newX, 0]])
            pts2 = np.float32([[0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst1 = cv2.warpPerspective(dst1, M, (cols, rows))

    return dst1




def get_angle(a,b,c):
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.array([int(qx), int(qy)])

def calculate_rotation(coords):
    (x1, y1) = coords[0]
    (x2, y2) = coords[1]
    if (x2-x1) == 0:
        return 90
    slope = (y2-y1)/(x2-x1)
    return math.degrees(math.atan(slope))


def crop_image(img,tol=0):
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]



def process(img,nr):
    h, w= img.shape[:2]
    center = (w // 2, h // 2)
    above = True

    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                                value=[0,0,0])


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    contToProcess = []
    maxLen = 0
    maxCont = None

    for cont in find_contours(gray, 6):

        if len(cont)>maxLen:
            maxCont = cont
            maxLen = len(cont)

    contToProcess.append(maxCont)

    for contour in contToProcess:

        coords = approximate_polygon(contour, tolerance=6)
        if len(coords)<6:
            coords = approximate_polygon(contour, tolerance=5)



        goodPoints = []

        if len(coords)>6:
            for i in range(1,len(coords)-1):
                if abs(get_angle(coords[i-1],coords[i],coords[i+1])-180)>10:
                    goodPoints.append(coords[i])

            if abs(get_angle(coords[-2], coords[0], coords[1]) - 180) > 10:
                goodPoints.append(coords[0])
            goodPoints.append(goodPoints[0])

            coords = np.array(goodPoints)


        points = list(zip(coords[:],coords[1:]))
        diff = list(map(lambda x : (x[0], (x[1][0][0]-x[1][1][0])**2+(x[1][0][1]-x[1][1][1])**2), enumerate(points)))
        diff.sort(key= lambda x : x[1],reverse=True)

        baseIndex = diff[0][0]
        idxToTest = baseIndex
        checkNext = False

        first= points[idxToTest][0]
        second =points[idxToTest][1]
        angle = calculate_rotation([first,second])
        firstTEST = rotate_point(center, (first[1], first[0]), angle * math.pi / 180)
        if firstTEST[1] < h // 2:
            angle +=180
        first = rotate_point(center, (first[1], first[0]), angle * math.pi / 180)

        coordsRotatedBase = list(map(lambda p: rotate_point(center, (p[1],p[0]), angle*math.pi/180),coords))
        coordsRotatedBase = np.array(coordsRotatedBase)

        countBad = 0
        for ic2 in range(len(coordsRotatedBase)-1):
            if coordsRotatedBase[ic2][0]>first[0]+10 and coordsRotatedBase[ic2+1][0]>first[0]+10:

                if not(coordsRotatedBase[ic2][1]<coordsRotatedBase[ic2+1][1]):
                    countBad+=1
                if countBad>2:
                    baseIndex = diff[1][0]
                    checkNext=True
                    break
        if checkNext:
            idxToTest = baseIndex

            first = points[idxToTest][0]
            second = points[idxToTest][1]
            angle = calculate_rotation([first, second])
            firstTEST = rotate_point(center, (first[1], first[0]), angle * math.pi / 180)
            if firstTEST[1] < h // 2:
                angle += 180
            first = rotate_point(center, (first[1], first[0]), angle * math.pi / 180)

            coordsRotatedBase = list(map(lambda p: rotate_point(center, (p[1], p[0]), angle * math.pi / 180), coords))
            coordsRotatedBase = np.array(coordsRotatedBase)

            countBad = 0
            for ic2 in range(len(coordsRotatedBase) - 1):
                if coordsRotatedBase[ic2][0] > first[0]+10 and coordsRotatedBase[ic2 + 1][0] > first[0]+10:
                    if not (coordsRotatedBase[ic2][1] < coordsRotatedBase[ic2 + 1][1]):
                        countBad += 1
                    if countBad > 2:
                        baseIndex = diff[2][0]

                        break


        refPoint = (0, 0)

        if baseIndex == len(diff) - 1:
            refPoint = points[baseIndex - 1][0]
        else:
            refPoint = points[baseIndex + 1][1]

        mask = []
        mask.append(list(points[baseIndex][0]))
        mask.append(list(points[baseIndex][1]))

        angle = calculate_rotation(mask)

        center = (w // 2, h // 2)

        basePoint = rotate_point((w // 2, h // 2), mask[0][::-1], angle * math.pi / 180)
        basePoint2 = rotate_point((w // 2, h // 2), mask[1][::-1], angle * math.pi / 180)
        mask = [list(basePoint), list(basePoint2)]

        coords2 = list(map(lambda p: rotate_point(center, (p[1], p[0]), angle * math.pi / 180), coords))
        minRef = 10000

        for c in coords2:
            cl = list(c)
            if cl in mask:
                continue
            if abs(c[0] - basePoint[0]) < minRef and abs(c[0] - basePoint[0]) > 20:
                minRef = abs(c[0] - basePoint[0])
                refPoint = c

        if basePoint[0] < refPoint[0]:
            above = False
        addToAngle = 90
        if not above:
            addToAngle = -90


        gray = rotate_bound(gray,angle+addToAngle)
        gray = crop_image(gray, 0)
        gray = gray[:-4]
        gray = unstreatch(gray)
        gray = crop_image(gray, 0)


        intervals = np.linspace(0,gray.shape[1],57,False, dtype=int)
        approximatedPoints = []
        for ix in intervals:
            approxH = np.min(np.argwhere(gray[:,ix]>0))
            approximatedPoints.append([approxH,ix])

        approximatedPoints= np.array(approximatedPoints)

    return normHeights(approximatedPoints[:, 0],nr)
