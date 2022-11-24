import numpy as np
from gd import f_analysis, f_analysis_time
from func import f1, f1prim, f1primprim
from func import func2, f2prim, f2primprim
from func import f3, f3prim, f3primprim
from func import black_f
from nelder_mead import nm
from scipy.spatial.distance import cdist


def fun1():
    point = np.array([0, 0, 0])
    _min = np.array([-1. / 6, -11. / 48, 1. / 6])
    x0 = np.array([point, [1, 0, 0], [0, 1, 0], [0, 0, 1]]) / 10
    x1 = np.array([
        point, [np.sqrt(2) - 1, 0, 0], [0, np.sqrt(2) - 1, 0],
        [0, 0, np.sqrt(2) - 1]
    ])
    x2 = np.array([
        point, [0.5 * np.sqrt(2), 0, 0], [0, 0.5 * np.sqrt(2), 0],
        [0, 0, 0.5 * np.sqrt(2)]
    ])
    x3 = np.array([
        point, [1.5 * np.sqrt(2), 0, 0], [0, 0.5 * np.sqrt(2), 0],
        [0, 0, 0.5 * np.sqrt(2)]
    ])
    x4 = 20 * np.sqrt(2) * x0
    print('Point proposal 0:', np.max(cdist(x0, x0)))
    x, y = nm(f1, x0)
    print('---------\n')
    print('Point proposal 1:', np.max(cdist(x1, x1)))
    x, y = nm(f1, x1)
    print('---------\n')

    print('Point proposal 2:', np.max(cdist(x2, x2)))
    x, y = nm(f1, x2)
    print('---------\n')

    print('Point proposal 3:', np.max(cdist(x3, x3)))
    x, y = nm(f1, x3)
    print('---------\n')

    print('Point proposal 4:', np.max(cdist(x4, x4)))
    x, y = nm(f1, x4)
    print('---------\n')

    print('\n---------------------\n F1 - GD: \n---------------------\n')
    for i in [2, 5, 10, 100]:
        print(f'For {i} steps:')
        f_analysis(f1, f1prim, f1primprim, point=point, _min=_min, st=i)
    for i in [0.1, 1, 2]:
        print(f'\nFor time of {i} sec.\n-----------------------\n')
        f_analysis_time(f1,
                        f1prim,
                        f1primprim,
                        point=point,
                        _min=_min,
                        _time=i)


def fun2():
    point = np.array([1.2, 1.2, 1.2])
    _min = np.array([1, 1, 1])
    x0 = np.array(
        [point, point + [1, 0, 0], point + [0, 1, 0], point + [0, 0, 1]]) / 10
    x1 = np.array([
        point, point + [np.sqrt(2) - 1, 0, 0],
        point + [0, np.sqrt(2) - 1, 0], point + [0, 0, np.sqrt(2) - 1]
    ])
    x2 = np.array([
        point, point + [0.5 * np.sqrt(2), 0, 0],
        point + [0, 0.5 * np.sqrt(2), 0], point + [0, 0, 0.5 * np.sqrt(2)]
    ])
    x3 = np.array([
        point, point + [1.5 * np.sqrt(2), 0, 0],
        point + [0, 0.5 * np.sqrt(2), 0], point + [0, 0, 0.5 * np.sqrt(2)]
    ])
    x4 = 20 * np.sqrt(2) * x0

    print('Point proposal 0:', np.max(cdist(x0, x0)))
    x, y = nm(func2, x0)
    print('---------\n')
    print('Point proposal 1:', np.max(cdist(x1, x1)))
    x, y = nm(func2, x1)
    print('---------\n')

    print('Point proposal 2:', np.max(cdist(x2, x2)))
    x, y = nm(func2, x2)
    print('---------\n')

    print('Point proposal 3:', np.max(cdist(x3, x3)))
    x, y = nm(func2, x3)
    print('---------\n')

    print('Point proposal 4:', np.max(cdist(x4, x4)))
    x, y = nm(func2, x4)
    print('---------\n')

    print('\n---------------------\n func2 - GD: \n---------------------\n')
    for i in [2, 5, 10, 100]:
        print(f'For {i} steps:')
        f_analysis(func2, f2prim, f2primprim, point=point, _min=_min, st=i)
    for i in [0.1, 1, 2]:
        print(f'\nFor time of {i} sec.\n-----------------------\n')
        f_analysis_time(func2,
                        f2prim,
                        f2primprim,
                        point=point,
                        _min=_min,
                        _time=i)


def fun3():
    point = np.array([1, 1])
    _min = np.array([3, 0.5])
    x0 = np.array([point, point + [1, 0], point + [0, 1]]) / 10
    x1 = np.array(
        [point, point + [np.sqrt(2) - 1, 0], point + [0, np.sqrt(2) - 1]])
    x2 = np.array(
        [point, point + [0.5 * np.sqrt(2), 0], point + [0, 0.5 * np.sqrt(2)]])
    x3 = np.array(
        [point, point + [1.5 * np.sqrt(2), 0], point + [0, 0.5 * np.sqrt(2)]])
    x4 = 20 * np.sqrt(2) * x0

    print('Point proposal 0:', np.max(cdist(x0, x0)))
    x, y = nm(f3, x0)
    print('---------\n')
    print('Point proposal 1:', np.max(cdist(x1, x1)))
    x, y = nm(f3, x1)
    print('---------\n')

    print('Point proposal 2:', np.max(cdist(x2, x2)))
    x, y = nm(f3, x2)
    print('---------\n')

    print('Point proposal 3:', np.max(cdist(x3, x3)))
    x, y = nm(f3, x3)
    print('---------\n')

    print('Point proposal 4:', np.max(cdist(x4, x4)))
    x, y = nm(f3, x4)
    print('---------\n')

    print('\n---------------------\n f3 - GD: \n---------------------\n')
    for i in [2, 5, 10, 100]:
        print(f'For {i} steps:')
        f_analysis(f3,
                   f3prim,
                   f3primprim,
                   point=point,
                   _min=_min,
                   st=i,
                   prj=lambda x, y: np.array([x, y]))
    for i in [0.1, 1, 2]:
        print(f'\nFor time of {i} sec.\n-----------------------\n')
        f_analysis_time(f3,
                        f3prim,
                        f3primprim,
                        point=point,
                        _min=_min,
                        _time=i,
                        prj=lambda x, y: np.array([x, y]))


def blackbox():

    point = np.array([0, 0, 0])
    x0 = np.array(
        [point, point + [1, 0, 0], point + [0, 1, 0], point + [0, 0, 1]]) / 10
    x1 = np.array([
        point, point + [np.sqrt(2) - 1, 0, 0],
        point + [0, np.sqrt(2) - 1, 0], point + [0, 0, np.sqrt(2) - 1]
    ])
    x2 = np.array([
        point, point + [0.5 * np.sqrt(2), 0, 0],
        point + [0, 0.5 * np.sqrt(2), 0], point + [0, 0, 0.5 * np.sqrt(2)]
    ])
    x3 = np.array([
        point, point + [1.5 * np.sqrt(2), 0, 0],
        point + [0, 0.5 * np.sqrt(2), 0], point + [0, 0, 0.5 * np.sqrt(2)]
    ])
    x4 = 20 * np.sqrt(2) * x0
    x4 = 20 * np.sqrt(2) * x0

    for i in range(1, 4):
        for _x in [x0, x1, x2, x3, x4]:
            print(f'x0:{_x} | Func(i={i})-------------\n')

            x, y = nm(lambda x: black_f(x, i), _x)
            print(f"i={i} x={x},y={y}")


if __name__ == '__main__':
    # fun1()
    # fun2()
    # fun3()
    blackbox()