import numpy as np

def normHeights(points):
    points = np.array(points)
    maxH = np.max(points)
    minH = np.min(points)
    points = map(lambda p : (p-minH)*200/(maxH-minH) , points)
    return np.array(list(points))

def vectorDiff(a, b):
    a = np.array(a)
    b = np.array(b)
    offset = len(a) - len(b)
    diffs = []
    for i in range(offset + 1):
        if i == 0:
            diff = a[offset:] - b
        else:
            diff = a[offset - i:-i] - b

        diffs.append(abs(np.sum(diff)))

    return diffs



def compareSet(setDict,truth):
    pairs = []
    for nr, vec in setDict.items():
        diffRank = []

        for i in setDict.keys():
            if i == nr:
                continue

            R = 2
            vec2 = np.array(setDict[i][::-1])
            vec2 = 200-vec2
            if len(setDict.keys())<30:
                d1 = np.sum((vec[3:] - vec2[3:])**R)
                d2 = np.sum((vec[3:] - vec2[:-3])**R)
                d3 = np.sum((vec[:-3] - vec2[3:])**R)
                d4 = np.sum((vec[:-3] - vec2[:-3])**R)
                d =  np.sum((vec - vec2)**R)

            else:
                vecA = normHeights(vec[3:])
                vecB = normHeights(vec[:-3])
                vec2A = normHeights(vec2[3:])
                vec2B = normHeights(vec2[:-3])
                d1 = np.sum((vecA - vec2A) ** R)
                d2 = np.sum((vecA - vec2B) ** R)
                d3 = np.sum((vecB - vec2A) ** R)
                d4 = np.sum((vecB - vec2B) ** R)
                d = np.sum((vec - vec2) ** R)
            minDiff = np.min([d1, d2, d3, d4, d])
            diffRank.append((i,minDiff))
        rankSorted = sorted(diffRank, key=lambda x: x[1])
        ranks = [str(x[0]) for x in rankSorted]
        if len(setDict.keys())>50:
            ranks = ranks[:30]
        print(" ".join(ranks))


    return pairs

