from __future__ import division

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

# TODO: functions that must GO!
# - fcross
# - edge
# - set pixel

class Vec3:
    def __init__(self, x, y, z=0):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def __repr__(self):
        return str((self.x, self.y, self.z))

class M3x3:
    def __init__(self, m11, m21, m31,
                       m12, m22, m32,
                       m13, m23, m33):
        self.m11 = float(m11)
        self.m21 = float(m21)
        self.m31 = float(m31)

        self.m12 = float(m12)
        self.m22 = float(m22)
        self.m32 = float(m32)

        self.m13 = float(m13)
        self.m23 = float(m23)
        self.m33 = float(m33)

    def __repr__(self):
        return str((self.m11, self.m12, self.m13)) + "\n" + \
               str((self.m12, self.m22, self.m32)) + "\n" + \
               str((self.m13, self.m23, self.m33)) + "\n"

class M4x4:
    def __init__(self, m11, m21, m31, m41,
                       m12, m22, m32, m42,
                       m13, m23, m33, m43,
                       m14, m24, m34, m44):

        self.m11 = float(m11)
        self.m21 = float(m21)
        self.m31 = float(m31)
        self.m41 = float(m41)

        self.m12 = float(m12)
        self.m22 = float(m22)
        self.m32 = float(m32)
        self.m42 = float(m42)

        self.m13 = float(m13)
        self.m23 = float(m23)
        self.m33 = float(m33)
        self.m43 = float(m43)

        self.m14 = float(m14)
        self.m24 = float(m24)
        self.m34 = float(m34)
        self.m44 = float(m44)

    def __repr__(self):
        return str((self.m11, self.m12, self.m13, self.m41)) + "\n" + \
               str((self.m12, self.m22, self.m32, self.m42)) + "\n" + \
               str((self.m13, self.m23, self.m33, self.m43)) + "\n" + \
               str((self.m14, self.m24, self.m34, self.m44)) + "\n"

def m3x3_mult(m, v, w):
    vx = m.m11 * v.x + m.m21 * v.y + m.m31 * w
    vy = m.m12 * v.x + m.m22 * v.y + m.m32 * w
    #vw = m.m13 * v.x + m.m23 * v.y + m.m33 * w
    return Vec3(vx, vy)

def m3x3_mult_with_w1(m, v):
    return m3x3_mult(m, v, 1.0)

def m3x3_concatenate(a, b):
    m = M3x3(0, 0, 0, 0, 0, 0, 0, 0, 0)

    m.m11 = a.m11 * b.m11 + a.m21 * b.m12 + a.m31 * b.m13
    m.m21 = a.m11 * b.m21 + a.m21 * b.m22 + a.m31 * b.m23
    m.m31 = a.m11 * b.m31 + a.m21 * b.m32 + a.m31 * b.m33

    m.m12 = a.m12 * b.m11 + a.m22 * b.m12 + a.m32 * b.m13
    m.m22 = a.m12 * b.m21 + a.m22 * b.m22 + a.m32 * b.m23
    m.m32 = a.m12 * b.m31 + a.m22 * b.m32 + a.m32 * b.m33

    m.m13 = a.m13 * b.m11 + a.m23 * b.m12 + a.m33 * b.m13
    m.m23 = a.m13 * b.m21 + a.m23 * b.m22 + a.m33 * b.m23
    m.m33 = a.m13 * b.m31 + a.m23 * b.m32 + a.m33 * b.m33

    return m

def m3x3_translate(tx, ty):
    m = M3x3(1, 0, tx, 0, 1, ty, 0, 0, 1)
    return m

def m3x3_rotate(theta):
    cost = math.cos(theta)
    sint = math.sin(theta)
    m = M3x3(cost, -sint, 0, sint, cost, 0, 0, 0, 1)
    return m

def m3x3_scale(sx, sy):
    m = M3x3(sx, 0, 0, 0, sy, 0, 0, 0, 1)
    return m

def m3x3_identity():
    return M3x3(1,0,0, 0,1,0, 0,0,1)

def m3x3_reflect_about_origin():
    return M3x3(-1,0,0, 0,-1,0, 0,0,1)

def m3x3_reflect_about_x_axis():
    return M3x3(1,0,0, 0,-1,0, 0,0,1)

def m3x3_reflect_about_y_axis():
    return M3x3(-1,0,0, 0,1,0, 0,0,1)

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
    v4 = Vec3(0, 0)

    if v0.y > v1.y: v0, v1 = v1, v0
    if v0.y > v2.y: v0, v2 = v2, v0
    if v1.y > v2.y: v1, v2 = v2, v1

    try:
        t = (v1.y - v0.y) / (v2.y - v0.y)
    except ZeroDivisionError:
        t = (v1.y - v0.y)

    v4.x = v0.x + t * (v2.x - v0.x)
    v4.y = v1.y

    try:
        m1 = (v2.x - v0.x) / (v2.y - v0.y)
        m2 = (v1.x - v0.x) / (v1.y - v0.y)
    except ZeroDivisionError:
        m1 = (v2.x - v0.x)
        m2 = (v1.x - v0.x)

    curra = v4.x
    currb = v1.x

    i = v4.y
    while (i > v0.y):
        line(data, int(curra), i, int(currb), i, color)
        curra -= m1
        currb -= m2
        i -= 1

    try:
        m1 = (v2.x - v4.x) / (v2.y - v4.y)
        m2 = (v2.x - v1.x) / (v2.y - v1.y)
    except ZeroDivisionError:
        m1 = (v2.x - v4.x)
        m2 = (v2.x - v1.x)


    curra = v4.x
    currb = v1.x

    i = v4.y
    while (i < v2.y):
        line(data, int(curra), i, int(currb), i, color)
        curra += m1
        currb += m2
        i += 1

def barycentric(a, b, c, p):
    try:
        u = float(((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / ((b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y)))
    except ZeroDivisionError:
        u = float((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y))
    try:
        v = float(((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / ((b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y)))
    except ZeroDivisionError:
        v = float((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y))
    w = 1.0 - u - v
    return (u, v, w)

def barycentric_raster_triangle(data, p0, p1, p2):
    if p0.y > p1.y: p0, p1 = p1, p0
    if p0.y > p2.y: p0, p2 = p2, p0
    if p1.y > p2.y: p1, p2 = p2, p1

    minx = min(p0.x, min(p1.x, p2.x))
    miny = min(p0.y, min(p1.y, p2.y))
    maxx = max(p0.x, max(p1.x, p2.x))
    maxy = max(p0.y, max(p1.y, p2.y))

    minx = max(minx, 0)
    miny = max(miny, 0)
    maxx = min(maxx, IMG_W-1)
    maxy = min(maxy, IMG_H-1)

    p = Vec3(minx, miny)
    while (p.y < maxy):
        # NOTE!  when copying C for loops, we have to think about the init. phase in our Python loops as well.
        if (p.x == maxx): p.x = minx
        while (p.x < maxx):
            lambda1, lambda2, lambda3 = barycentric(p0, p1, p2, p)
            if ((lambda1 >= 0.0) and (lambda2 >= 0.0) and (lambda3 >= 0.0)):
                c = (chr(int(lambda1 * 255)), chr(int(lambda2 * 255)), chr(int(lambda3 * 255)))
                set_pixel(data, int(p.x), int(p.y), c)
            p.x += 1
        p.y += 1

def fcross(a, b):
    cross = Vec3(0, 0, 0)
    cross.x = float(a.y * b.z - a.z * b.y)
    cross.y = float(a.z * b.x - a.x * b.z)
    cross.z = float(a.x * b.y - a.y * b.x)
    return cross

def lerp(a, t, b):
    if t > 1 or t < 0: return 0
    return (1.0 - t) * a + t * b

def bilinear(tx, ty, c00, c10, c01, c11):
    b = lerp(c00, tx, c10)
    a = lerp(c10, tx, c11)
    return lerp(a, ty, b)

def edge(a, b, c):
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)

def degrees_to_radiants(d):
    return (d * math.pi) / 180

def radiants_to_degrees(r):
    return (r * 180) / math.pi

# NOTE
# ~~~~
# this might have some corner cases where it is not working properly
# it would be nice to see it in a live action scenario
# maybe we could create a list of rotating images?
# ~~~~

# NOTE: bytearrays ARE BAD ~!

def show_bounding_box(data, t):
    line(data, t[0].x, t[0].y, t[0].x, t[1].y, WHITE)
    line(data, t[0].x, t[1].y, t[2].x, t[1].y, WHITE)
    line(data, t[2].x, t[1].y, t[2].x, t[2].y, WHITE)
    line(data, t[0].x, t[0].y, t[2].x, t[2].y, WHITE)

def find_centroid(t):
    tx = t[0].x + t[1].x + t[2].x
    ty = t[0].y + t[1].y + t[2].y
    return (int(tx / 3), int(ty / 3))

def write_tga(filename, data):
        HEADER = struct.pack("<3b2hb4h2b", 0, 0, IMG_DSC_NON_RLE, 0, 0, 0, 0, 0, IMG_W, IMG_H, 24, 0)
        FOOTER = bytearray(8) + bytearray(b"TRUEVISION-.xFILE.") + bytearray(1)
        f = open(filename, "wb")
        f.write(HEADER)
        f.write("".join(data))
        f.write(FOOTER)
        f.close()

def load_non_rle_texture():
    f = open("obj/non_rle_texture.tga", "rb")
    f.seek(18,0)
    data = bytearray(f.read())
    f.close()
    return data

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
    p0 = Vec3(0, 0, 0)
    p1 = Vec3(0, 0, 0)
    p2 = Vec3(0, 0, 0)

    d1 = Vec3(0, 0, 0)
    d2 = Vec3(0, 0, 0)

    for i in xrange(N_FACES):
        v0 = objdata['v'][objdata['f'][i][0]-1]
        v1 = objdata['v'][objdata['f'][i][3]-1]
        v2 = objdata['v'][objdata['f'][i][6]-1]

        vt0 = objdata['vt'][objdata['f'][i][1]-1]
        vt1 = objdata['vt'][objdata['f'][i][4]-1]
        vt2 = objdata['vt'][objdata['f'][i][7]-1]

        #uv0 = Vec3(vt0[0] * 1023, vt0[1] * 1023)
        #uv1 = Vec3(vt1[0] * 1023, vt1[1] * 1023)
        #uv2 = Vec3(vt2[0] * 1023, vt2[1] * 1023)

        #uv0 = Vec3(vt0[0], vt0[1])
        #uv1 = Vec3(vt1[0], vt1[1])
        #uv2 = Vec3(vt2[0], vt2[1])

        p0.x = int((v0[0] + 1) * IMG_W/2)
        p0.y = int((v0[1] + 1) * IMG_H/2)
        p1.x = int((v1[0] + 1) * IMG_W/2)
        p1.y = int((v1[1] + 1) * IMG_H/2)
        p2.x = int((v2[0] + 1) * IMG_W/2)
        p2.y = int((v2[1] + 1) * IMG_H/2)

        # NOTE: we should have the vertex normal information in our obj.
        # I think that we don't need to calculate everything ourselves. We could just parse for that data.
        d1.x = v1[0] - v0[0]
        d1.y = v1[1] - v0[1]
        d1.z = v1[2] - v0[2]

        d2.x = v2[0] - v0[0]
        d2.y = v2[1] - v0[1]
        d2.z = v2[2] - v0[2]

        normal = fcross(d1, d2)

        dist = math.sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z)

        normal.x = normal.x / dist
        normal.y = normal.y / dist
        normal.z = normal.z / dist

        direction = (normal.x * 0) + (normal.y * 0) + (normal.z * 1)

        if (direction > 0):

            minx = min(p0.x, min(p1.x, p2.x))
            miny = min(p0.y, min(p1.y, p2.y))
            maxx = max(p0.x, max(p1.x, p2.x))
            maxy = max(p0.y, max(p1.y, p2.y))

            minx = max(minx, 0)
            miny = max(miny, 0)
            maxx = min(maxx, IMG_W-1)
            maxy = min(maxy, IMG_H-1)

            p = Vec3(minx, miny)
            while (p.y < maxy):
                # NOTE!  when copying C for loops, we have to think about the init. phase in our Python loops as well.
                if (p.x == maxx): p.x = minx
                while (p.x < maxx):
                    w0 = edge(p1, p2, p)
                    w1 = edge(p2, p0, p)
                    w2 = edge(p0, p1, p)

                    if ((w0 >= 0) and (w1 >= 0) and (w2 >= 0)):
                        p.z = 0.0
                        p.z += v0[2] * p.x + p.y * 3
                        p.z += v1[2] * p.x + p.y * 3
                        p.z += v2[2] * p.x + p.y * 3

                        if (zbuffer[p.x + p.y * IMG_W] < p.z):
                            zbuffer[p.x + p.y * IMG_W] = p.z
                            bgr = (chr(int(direction * 255)), chr(int(direction * 255)), chr(int(direction * 255)))
                            set_pixel(data, p.x, p.y, bgr)
                    p.x += 1
                p.y += 1

def rle_encode(data):
    run = 1
    byte = 0
    maxrun = 128
    rle_buffer = bytearray()
    rle_current_byte = 0
    while byte < IMG_SIZE:
        while byte < IMG_SIZE and run < maxrun and data[byte:byte+3] == data[byte+3:byte+3+3]:
            byte += 3
            if data[byte:byte+3] == data[byte+3:byte+3+3]:
                run += 1
            if data[byte:byte+3] != data[byte+3:byte+3+3]:
                byte += 3
                run ^= 1 << 7
                rle_buffer += bytearray(5)
                rle_buffer[rle_current_byte] = run
                rle_buffer[rle_current_byte+1] = data[byte-3]
                rle_buffer[rle_current_byte+2] = data[byte-2]
                rle_buffer[rle_current_byte+3] = data[byte-1]
                run = 1
        if data[byte:byte+3] == data[byte+3:byte+3+3]:
            if run == maxrun:
                run -= 1
                run ^= 1 << 7
                print byte
                rle_buffer[byte] = run
                rle_buffer[byte+1] = data[byte-3]
                rle_buffer[byte+2] = data[byte-2]
                rle_buffer[byte+3] = data[byte-1]
                run = 1
        while byte < IMG_SIZE and run < maxrun and data[byte:byte+3] != data[byte+3:byte+3+3]:
            byte += 3
            if data[byte:byte+3] != data[byte+3:byte+3+3]:
                run += 1
            if data[byte:byte+3] == data[byte+3:byte+3+3]:
                if run == 1:
                    run = 0
                    rle_buffer[byte] = run
                    rle_buffer[byte+1] = data[byte-3]
                    rle_buffer[byte+2] = data[byte-2]
                    rle_buffer[byte+3] = data[byte-1]
                    run = 1
                    break
                else:
                    run -= 1
                    rle_buffer[byte] = run
                    rle_buffer[byte+1] = data[byte-3]
                    rle_buffer[byte+2] = data[byte-2]
                    rle_buffer[byte+3] = data[byte-1]
                    run = 1
                    break
        if data[byte:byte+3] != data[byte+3:byte+3+3]:
            if run == maxrun:
                run -= 1
                rle_buffer[byte] = run
                rle_buffer[byte+1] = data[byte-3]
                rle_buffer[byte+2] = data[byte-2]
                rle_buffer[byte+3] = data[byte-1]
                run = 1
    return rle_buffer

def main():
    DATA = [chr(0)] * IMG_SIZE
    ZBUFFER = [0]*IMG_SIZE
    OBJ_DATA = {"v": [], "vt": [], "f": []}

    parse_obj(OBJ_DATA)
    #texture = load_non_rle_texture()

    t = [Vec3(200,100), Vec3(600,600), Vec3(800,100)]
    centroid = find_centroid(t)
    m = m3x3_identity()
    m = m3x3_concatenate(m, m3x3_translate(centroid[0], centroid[1]))
    m = m3x3_concatenate(m, m3x3_rotate(degrees_to_radiants(30)))
    m = m3x3_concatenate(m, m3x3_translate(-centroid[0], -centroid[1]))
    t[0] = m3x3_mult_with_w1(m, t[0])
    t[1] = m3x3_mult_with_w1(m, t[1])
    t[2] = m3x3_mult_with_w1(m, t[2])
    barycentric_raster_triangle(DATA, t[0], t[1], t[2])
    show_bounding_box(DATA, t)

    construct_model(DATA, ZBUFFER, OBJ_DATA)

    # TODO: NOT IMPLEMENTED!  #RLE_DATA = rle_encode(DATA[0:(int(IMG_SIZE/2))])
    write_tga("test_rle.tga", DATA)

if __name__ == "__main__":
    main()
