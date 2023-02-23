# import multiprocessing
# from multiprocessing import Pool
# import time
#
#
# def f(x):
#     return pow(x, x, 117)
#
#
# def k(x):
#     return pow(x, x, 117)
#
#
# if __name__ == '__main__':
#     n_cpu = multiprocessing.cpu_count()
#     # multiprocessing:
#     print(f'logic cores number: {n_cpu}')
#     p = Pool(processes=16)
#
#     # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
#     start = time.time_ns()
#     results = p.map(f, [i for i in range(10000000)])
#     # print(p.map(f, [1, 2, 3]))
#     finish = time.time_ns()
#     print(f"It's done!")
#     print(f'time elapsed: {(finish - start) // 10 ** 6} ms')
#     # serial:
#     a = map(k, [i for i in range(10000000)])
#     end = time.time_ns()
#     print(f"It's done!")
#     print(f'time elapsed: {(end - finish) // 10 ** 6} ms')

import arcade

# start = time.time_ns()
# for i in range(20500):
#     sprite = arcade.Sprite('right-arrow-black-triangle.png', 0.125,  center_x=1000, center_y=500)
#     sprite.angle = 90
# finish = time.time_ns()
# print(f'time elapsed: {(finish - start) // 10 ** 6} ms')



