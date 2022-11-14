# pyrenderer

Pythonic rendering experiments.

Abandoned due to python being too slow on my machine. This was useful for me to understand just
how slow python can be. Waiting 18s where C takes 1s.

Here are some benchmark results:

`python
    test_set_single_pixel 0.19697397399999997
    test_set_myltiple_pixels 0.19873901400000005

    int_edge 0.6665156689999999
    int_edge_fn_call 0.6662710380000001
    list_edge 1.180054629
    list_edge_fn_call 1.1774959630000001
    tuple_edge 1.1248761740000006
    class_edge 1.3790133850000004
    slots_class_edge 1.162511619

    deque_edge 1.3116176129999992
    slow_list_append 0.9250420100000003
    fast_deque_append 0.7129976439999997

    slow_sqrt 0.5451525060000009
    fast_sqrt 0.9073496599999995

    slow_fcross 12.982337518
    fast_fcross 3.580524406000002

    if_speed_test 0.8057347429999986
    try_speed_test 0.37435757800000147

    slow_setpixel 1.9537633660000004
    fast_setpixel 0.9951066739999987
`

All benchmarks were run on a ASUS E200HA laptop.
