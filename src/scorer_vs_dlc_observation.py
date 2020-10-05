
import numpy as np

def direction(p1 = None, p3= None, p4= None):
    d = np.cross((p1-p3),(p4-p3))
    return np.sign(d)


# checks if line segment p1p2 and p3p4 intersect
def intersect(p1, p2, p3, p4):
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
        ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    else:
        return False

    #elif d1 == 0 and on_segment(p3, p4, p1):
    #    return True
    #elif d2 == 0 and on_segment(p3, p4, p2):
    #    return True
    #elif d3 == 0 and on_segment(p1, p2, p3):
    #    return True
    #elif d4 == 0 and on_segment(p1, p2, p4):
    #    return True
    #else:
        #return False