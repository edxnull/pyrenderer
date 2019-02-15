from __future__ import division

import pdb
import time
import math
import struct

IMG_W = 1024
IMG_H = 1024
IMG_BPP = 3
IMG_DSC_RLE = 10
IMG_DSC_NON_RLE = 2
IMG_SIZE = IMG_W * IMG_H * IMG_BPP

TEXTURE_IMG_W = 1024
TEXTURE_IMG_H = 1024

# NOTE: These colors are BGR not RGB!
RED    = (chr(0x00), chr(0x00), chr(0xff))
BLUE   = (chr(0xff), chr(0x00), chr(0x00))
CYAN   = (chr(0xff), chr(0xff), chr(0x00))
GREEN  = (chr(0x00), chr(0xff), chr(0x00))
WHITE  = (chr(0xff), chr(0xff), chr(0xff))
BLACK  = (chr(0x00), chr(0x00), chr(0x00))
YELLOW = (chr(0x00), chr(0xff), chr(0xff))
VIOLET = (chr(0xff), chr(0x00), chr(0xff))

N_VERTS = 1258
N_FACES = 2492
N_VT    = 1339

def m3x3_mult(m, v, w):
    m11, m21, m31, m12, m22, m32, m13, m23, m33 = 0, 1, 2, 3, 4, 5, 6, 7, 8
    vx = m[m11] * v[0] + m[m21] * v[1] + m[m31] * w
    vy = m[m12] * v[0] + m[m22] * v[1] + m[m32] * w
    return [int(vx), int(vy), 0]

def m3x3_mult_with_w1(m, v):
    return m3x3_mult(m, v, 1.0)

def m3x3_concatenate(a, b):
    m = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    m11, m21, m31 = 0, 1, 2
    m12, m22, m32 = 3, 4, 5
    m13, m23, m33 = 5, 6, 7

    m[m11] = a[m11] * b[m11] + a[m21] * b[m12] + a[m31] * b[m13]
    m[m21] = a[m11] * b[m21] + a[m21] * b[m22] + a[m31] * b[m23]
    m[m31] = a[m11] * b[m31] + a[m21] * b[m32] + a[m31] * b[m33]

    m[m12] = a[m12] * b[m11] + a[m22] * b[m12] + a[m32] * b[m13]
    m[m22] = a[m12] * b[m21] + a[m22] * b[m22] + a[m32] * b[m23]
    m[m32] = a[m12] * b[m31] + a[m22] * b[m32] + a[m32] * b[m33]

    m[m13] = a[m13] * b[m11] + a[m23] * b[m12] + a[m33] * b[m13]
    m[m23] = a[m13] * b[m21] + a[m23] * b[m22] + a[m33] * b[m23]
    m[m33] = a[m13] * b[m31] + a[m23] * b[m32] + a[m33] * b[m33]

    return map(float, m)

def m3x3_translate(tx, ty):
    return map(float, [1, 0, tx, 0, 1, ty, 0, 0, 1])

def m3x3_rotate(theta):
    cost = math.cos(theta)
    sint = math.sin(theta)
    return map(float, [cost, -sint, 0, sint, cost, 0, 0, 0, 1])

def m3x3_scale(sx, sy):
    return map(float, [sx, 0, 0, 0, sy, 0, 0, 0, 1])

def m3x3_identity():
    return map(float, [1, 0, 0, 0, 1, 0, 0, 0, 1])

def m3x3_reflect_about_origin():
    return map(float, [-1,0,0, 0,-1,0, 0,0,1])

def m3x3_reflect_about_x_axis():
    return map(float, [1,0,0, 0,-1,0, 0,0,1])

def m3x3_reflect_about_y_axis():
    return map(float, [-1,0,0, 0,1,0, 0,0,1])

def set_pixel(data, x, y, color):
    s = (x + y * IMG_W) * IMG_BPP
    data[s:s+3] = color

def line(data, x0, y0, x1, y1, color):
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = int((dx if dx > dy else -dy) / 2)
    e2 = int()

    while True:
        set_pixel(data, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = err
        if (e2 > -dx):
            err -= dy
            x0 += sx
        if (e2 < dy):
            err += dx
            y0 += sy

def triangle(data, v0, v1, v2, color):
    v4 = [0,0,0]

    X, Y, Z = 0, 1, 2

    if v0[Y] > v1[Y]: v0, v1 = v1, v0
    if v0[Y] > v2[Y]: v0, v2 = v2, v0
    if v1[Y] > v2[Y]: v1, v2 = v2, v1

    try:
        t = (v1[Y] - v0[Y]) / (v2[Y] - v0[Y])
    except ZeroDivisionError:
        print "exception got triggered"
        t = (v1[Y] - v0[Y])

    v4[X] = v0[X] + t * (v2[X] - v0[X])
    v4[Y] = v1[Y]

    try:
        m1 = (v2[X] - v0[X]) / (v2[Y] - v0[Y])
        m2 = (v1[X] - v0[X]) / (v1[Y] - v0[Y])
    except ZeroDivisionError:
        print "exception got triggered"
        m1 = (v2[X] - v0[X])
        m2 = (v1[X] - v0[X])

    curra = v4[X]
    currb = v1[X]

    i = v4[Y]
    while (i > v0[Y]):
        line(data, int(curra), i, int(currb), i, color)
        curra -= m1
        currb -= m2
        i -= 1

    try:
        m1 = (v2[X] - v4[X]) / (v2[Y] - v4[Y])
        m2 = (v2[X] - v1[X]) / (v2[Y] - v1[Y])
    except ZeroDivisionError:
        print "exception got triggered"
        m1 = (v2[X] - v4[X])
        m2 = (v2[X] - v1[X])


    curra = v4[X]
    currb = v1[X]

    i = v4[Y]
    while (i < v2[Y]):
        line(data, int(curra), i, int(currb), i, color)
        curra += m1
        currb += m2
        i += 1


#TODO: TEST!!!!
def faster_barycentric(a0, a1, b0, b1, c0, c1, p0, p1):
    diff = (b1 - c1)*(a0 - c0) + (c0 - b0)*(a1 - c1)
    u = float(((b1 - c1) * (p0 - c0) + (c0 - b0) * (p1 - c1)) / diff)
    v = float(((c1 - a1) * (p0 - c0) + (a0 - c0) * (p1 - c1)) / diff)
    w = 1.0 - u - v
    return (u, v, w)

def barycentric(a, b, c, p):
	u = float(((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])))
	v = float(((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])))
	w = 1.0 - u - v
	return (u, v, w)

def barycentric_raster_triangle(data, p0, p1, p2):
    if p0[1] > p1[1]: p0, p1 = p1, p0
    if p0[1] > p2[1]: p0, p2 = p2, p0
    if p1[1] > p2[1]: p1, p2 = p2, p1

    minx = min(p0[0], min(p1[0], p2[0]))
    miny = min(p0[1], min(p1[1], p2[1]))
    maxx = max(p0[0], max(p1[0], p2[0]))
    maxy = max(p0[1], max(p1[1], p2[1]))

    minx = max(minx, 0)
    miny = max(miny, 0)
    maxx = min(maxx, IMG_W-1)
    maxy = min(maxy, IMG_H-1)

    p = [minx, miny, 0]
    while (p[1] < maxy):
        if (p[0] == maxx): p[0] = minx
        while (p[0] < maxx):
            #lambda1, lambda2, lambda3 = barycentric(p0, p1, p2, p)
            lambda1, lambda2, lambda3 = faster_barycentric(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], p[0], p[1])
            if ((lambda1 >= 0.0) and (lambda2 >= 0.0) and (lambda3 >= 0.0)):
                color = (chr(int(lambda1 * 255)), chr(int(lambda2 * 255)), chr(int(lambda3 * 255)))
                set_pixel(data, int(p[0]), int(p[1]), color)
            p[0] += 1
        p[1] += 1

def fcross(a, b):
    cross = [0, 0, 0]
    X, Y, Z = 0, 1, 2
    cross[0] = float(a[Y] * b[Z] - a[Z] * b[Y])
    cross[1] = float(a[Z] * b[X] - a[X] * b[Z])
    cross[2] = float(a[X] * b[Y] - a[Y] * b[X])
    return cross

def lerp(a, t, b):
    if t > 1 or t < 0: return 0
    return (1.0 - t) * a + t * b

def bilinear(tx, ty, c00, c10, c01, c11):
    b = lerp(c00, tx, c10)
    a = lerp(c10, tx, c11)
    return lerp(a, ty, b)

# NOTE
# I wonder if it would be possible to speed it up by passing a0, a1, b0, b1, c0, c1 instead of a, b, c
# We've gained a 1s!
def faster_edge(a0, a1, b0, b1, c0, c1):
    return (b0 - a0)*(c1 - a1) - (b1 - a1)*(c0 - a0)

def edge(a, b, c):
    return (b[0] - a[0])*(c[1] - a1) - (b[1] - a1)*(c[0] - a[0])

def degrees_to_radiants(d):
    return (d * math.pi) / 180

def radiants_to_degrees(r):
    return (r * 180) / math.pi

def show_bounding_box(data, t):
    line(data, t[0][0], t[0][1], t[0][0], t[1][1], WHITE)
    line(data, t[0][0], t[1][1], t[2][0], t[1][1], WHITE)
    line(data, t[2][0], t[1][1], t[2][0], t[2][1], WHITE)
    line(data, t[0][0], t[0][1], t[2][0], t[2][1], WHITE)

def find_centroid(t):
    tx = t[0][0] + t[1][0] + t[2][0]
    ty = t[0][1] + t[1][1] + t[2][1]
    return [int(tx / 3), int(ty / 3)]

def write_tga(filename, data):
    HEADER = struct.pack("<3b2hb4h2b", 0, 0, IMG_DSC_NON_RLE, 0, 0, 0, 0, 0, IMG_W, IMG_H, 24, 0)
    FOOTER = bytearray(8) + bytearray(b"TRUEVISION-.xFILE.") + bytearray(1)
    f = open(filename, "wb")
    f.write(HEADER)
    f.write("".join(data))
    f.write(FOOTER)
    f.close()

def parse_obj(objdata):
    f = open("obj/african_head/african_head.obj", "r")
    data = f.read()
    f.close()
    data = data.split("\n")
    for line in xrange(len(data)):
        split_data = data[line].split(" ")
        if split_data[0] == 'v':
            objdata['v'].append([float(element) for element in split_data[1::]])
        if split_data[0] == 'vt':
            objdata['vt'].append([float(element) for element in split_data[2::]])
        if split_data[0] == 'f':
            t = []
            for i in xrange(3):
                for element in split_data[i+1].split("/"):
                    t.append(int(element))
            objdata['f'].append(t)

def wireframe_render_model(data, objdata, color=WHITE):
    p0 = Vec3(0, 0, 0)
    p1 = Vec3(0, 0, 0)
    p2 = Vec3(0, 0, 0)
    for i in xrange(N_FACES):
        v0 = objdata['v'][objdata['f'][i][0]-1]
        v1 = objdata['v'][objdata['f'][i][3]-1]
        v2 = objdata['v'][objdata['f'][i][6]-1]

        p0.x = int((v0[0] + 1) * IMG_W/2)
        p0.y = int((v0[1] + 1) * IMG_H/2)
        p1.x = int((v1[0] + 1) * IMG_W/2)
        p1.y = int((v1[1] + 1) * IMG_H/2)
        p2.x = int((v2[0] + 1) * IMG_W/2)
        p2.y = int((v2[1] + 1) * IMG_H/2)

        line(data, p0.x, p0.y, p1.x, p1.y, color)
        line(data, p1.x, p1.y, p2.x, p2.y, color)
        line(data, p2.x, p2.y, p0.x, p0.y, color)

def construct_model(data, zbuffer, objdata, texture=None):
    p0 = [0, 0, 0]
    p1 = [0, 0, 0]
    p2 = [0, 0, 0]

    d1 = [0, 0, 0]
    d2 = [0, 0, 0]

    X, Y, Z = 0, 1, 2

    HALF_OF_IMG_W = IMG_W/2
    HALF_OF_IMG_H = IMG_H/2

    for i in xrange(N_FACES):
        v0 = objdata['v'][objdata['f'][i][0]-1]
        v1 = objdata['v'][objdata['f'][i][3]-1]
        v2 = objdata['v'][objdata['f'][i][6]-1]

        p0[0] = int((v0[0] + 1) * HALF_OF_IMG_W)
        p0[1] = int((v0[1] + 1) * HALF_OF_IMG_H)
        p1[0] = int((v1[0] + 1) * HALF_OF_IMG_W)
        p1[1] = int((v1[1] + 1) * HALF_OF_IMG_H)
        p2[0] = int((v2[0] + 1) * HALF_OF_IMG_W)
        p2[1] = int((v2[1] + 1) * HALF_OF_IMG_H)

        d1[0] = v1[0] - v0[0]
        d1[1] = v1[1] - v0[1]
        d1[2] = v1[2] - v0[2]

        d2[0] = v2[0] - v0[0]
        d2[1] = v2[1] - v0[1]
        d2[2] = v2[2] - v0[2]

        normal = fcross(d1, d2)

        dist = math.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])

        normal[0] = normal[0] / dist
        normal[1] = normal[1] / dist
        normal[2] = normal[2] / dist

        direction = (normal[0] * 0) + (normal[1] * 0) + (normal[2] * 1)

        if (direction > 0):

            minx = min(p0[X], min(p1[X], p2[X]))
            miny = min(p0[Y], min(p1[Y], p2[Y]))
            maxx = max(p0[X], max(p1[X], p2[X]))
            maxy = max(p0[Y], max(p1[Y], p2[Y]))

            minx = max(minx, 0)
            miny = max(miny, 0)
            maxx = min(maxx, IMG_W-1)
            maxy = min(maxy, IMG_H-1)

            p = [minx, miny, 0]
            #NOTE: @SLOW!
            while (p[Y] < maxy):
            #for i in range(p[Y], maxy):
                if (p[X] == maxx): p[X] = minx
                while (p[X] < maxx):
                #for j in range(p[X], maxx):
                    # p0, p1, p2 do not change here
                    # NOTE: what if we don't save w0, w1, w2??? can we gain speed?
                    if faster_edge(p1[0], p1[1], p2[0], p2[1], p[0], p[1]) >= 0:
                        if faster_edge(p2[0], p2[1], p0[0], p0[1], p[0], p[1]) >= 0:
                            if faster_edge(p0[0], p0[1], p1[0], p1[1], p[0], p[1]) >= 0:
                            #if (faster_edge(p1[0], p1[1], p2[0], p2[1], p[0], p[1]) >= 0) and \
                            #   (faster_edge(p2[0], p2[1], p0[0], p0[1], p[0], p[1]) >= 0) and \
                            #   (faster_edge(p0[0], p0[1], p1[0], p1[1], p[0], p[1]) >= 0):
                                p[Z] = 0.0
                                p[Z] += v0[2] * p[X] + p[Y] * 3
                                p[Z] += v1[2] * p[X] + p[Y] * 3
                                p[Z] += v2[2] * p[X] + p[Y] * 3

                                if (zbuffer[p[X] + p[Y] * IMG_W] < p[Z]):
                                    zbuffer[p[X] + p[Y] * IMG_W] = p[Z]
                                    #nd = direction * 255
                                    #rgb = chr(int(nd))
                                    #bgr = (rgb, rgb, rgb)
                                    clr = chr(int(direction*255))
                                    set_pixel(data, p[X], p[Y], (clr, clr, clr))
                    p[X] += 1
                p[Y] += 1

def main():
    now = time.time()

    DATA = [chr(0)]*IMG_SIZE
    ZBUFFER = [0]*IMG_SIZE
    OBJ_DATA = {'v' : [], 'vt': [], 'f' : []}

    parse_obj(OBJ_DATA)

    #t = [[200,300,0], [400,700,0], [800,300,0]]
    #barycentric_raster_triangle(DATA, t[0], t[1], t[2])
    construct_model(DATA, ZBUFFER, OBJ_DATA)
    #centroid = find_centroid(t)
    #m = m3x3_identity()
    #m = m3x3_concatenate(m, m3x3_translate(centroid[0], centroid[1]))
    #m = m3x3_concatenate(m, m3x3_rotate(degrees_to_radiants(90)))
    #m = m3x3_concatenate(m, m3x3_translate(-centroid[0], -centroid[1]))
    #t[0] = m3x3_mult_with_w1(m, t[0])
    #t[1] = m3x3_mult_with_w1(m, t[1])
    #t[2] = m3x3_mult_with_w1(m, t[2])
    #barycentric_raster_triangle(DATA, t[0], t[1], t[2])
    #show_bounding_box(DATA, t)

    # TODO: NOT IMPLEMENTED!  #RLE_DATA = rle_encode(DATA[0:(int(IMG_SIZE/2))])
    write_tga("test_rle.tga", DATA)

    end = time.time()
    print "ellapsed time " + str((end - now))

if __name__ == "__main__":
    main()
