import timeit
from collections import deque

IMG_W = 1024
IMG_H = 1024
IMG_BPP = 3
IMG_DSC_RLE = 10
IMG_DSC_NON_RLE = 2
IMG_SIZE = IMG_W * IMG_H * IMG_BPP
GREEN  = (chr(0x00), chr(0xff), chr(0x00))

class Vec3:
    def __init__(self, x, y, z=0):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def __repr__(self):
        return str((self.x, self.y, self.z))

class Foo:
    __slots__ = ("x", "y", "z")
    def __init__(self, x, y, z=0):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def __repr__(self):
        return str((self.x, self.y, self.z))

TIMES = 999999

def tuple_edge():
    a = tuple((2,7,2))
    b = tuple((2,7,2))
    c = tuple((2,7,2))
    for x in xrange(TIMES):
        temp = (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])

def class_edge():
    a = Vec3(2, 7, 2)
    b = Vec3(2, 7, 2)
    c = Vec3(2, 7, 2)
    for x in xrange(TIMES):
        temp = (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)

def slots_class_edge():
    a = Foo(2, 7, 2)
    b = Foo(2, 7, 2)
    c = Foo(2, 7, 2)
    for x in xrange(TIMES):
        temp = (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)

def list_edge():
    a = [2,7,2]
    b = [2,7,2]
    c = [2,7,2]
    for x in xrange(TIMES):
        temp = (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])

def deque_edge():
    a = deque([2,7,2])
    b = deque([2,7,2])
    c = deque([2,7,2])
    for x in xrange(TIMES):
        temp = (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])

def slow_setpixel():
    x = 100
    y = 100
    DATA = bytearray(IMG_SIZE)
    for time in xrange(TIMES):
        s = (x + y * IMG_W) * IMG_BPP
        DATA[s:s+3] = GREEN

def fast_setpixel():
    x = 100
    y = 100
    DATA = [chr(0)] * IMG_SIZE
    for time in xrange(TIMES):
        s = (x + y * IMG_W) * IMG_BPP
        DATA[s:s+3] = GREEN
    result = "".join(DATA)
    #print repr(result[0:10])

def slow_list_append():
    a = []
    for time in xrange(TIMES*2):
        a.append(time)

def fast_deque_append():
    a = deque([])
    for time in xrange(TIMES*2):
        a.append(time)

def slow_sqrt():
    import math
    for x in xrange(TIMES):
        dist = math.sqrt(0.9 * 0.9 * 0.9 * 0.9 * 0.8 * 0.1)

def fast_sqrt():
    import cmath
    for x in xrange(TIMES):
        dist = cmath.sqrt(0.9 * 0.9 * 0.9 * 0.9 * 0.8 * 0.1)

def slow_fcross():
    for time in xrange(TIMES):
        cross = Vec3(0, 0, 0)
        a = Vec3(8,3,4)
        b = Vec3(3,3,4)
        cross.x = float(a.y * b.z - a.z * b.y)
        cross.y = float(a.z * b.x - a.x * b.z)
        cross.z = float(a.x * b.y - a.y * b.x)

def fast_fcross():
    for time in xrange(TIMES):
        cross = [0, 0, 0]
        a = [8,3,4]
        b = [3,3,4]
        cross[0] = float(a[1] * b[2] - a[2] * b[1])
        cross[1] = float(a[2] * b[0] - a[0] * b[2])
        cross[2] = float(a[0] * b[1] - a[1] * b[0])

def if_speed_test():
    a = 0
    for time in xrange(TIMES*3):
        if (1 > 0) and (2 > 0):
            a = 1 + 1
        else:
            a = 1 + 1

def try_speed_test():
    a = 0
    for time in xrange(TIMES*3):
        try:
            a = 1+1
        except ZeroDivisionError:
            a = 1+1

def main():
    print "tuple_edge",   timeit.timeit(tuple_edge, number=1)
    print "class_edge",  timeit.timeit(class_edge, number=1)
    print "slots_class_edge",  timeit.timeit(slots_class_edge, number=1)
    print "list_edge",  timeit.timeit(list_edge, number=1)
    print "deque_edge",  timeit.timeit(deque_edge, number=1)
    print "slow_setpixel",  timeit.timeit(slow_setpixel, number=1)
    print "fast_setpixel",  timeit.timeit(fast_setpixel, number=1)
    print "slow_list_append",  timeit.timeit(slow_list_append, number=1)
    print "fast_deque_append",  timeit.timeit(fast_deque_append, number=1)
    print "slow_sqrt",  timeit.timeit(slow_sqrt, number=1)
    print "fast_sqrt",  timeit.timeit(fast_sqrt, number=1)
    print "slow_fcross",  timeit.timeit(slow_fcross, number=1)
    print "fast_fcross",  timeit.timeit(fast_fcross, number=1)
    print "if_speed_test", timeit.timeit(if_speed_test, number=1)
    print "try_speed_test", timeit.timeit(try_speed_test, number=1)

if __name__ == "__main__":
    main()
