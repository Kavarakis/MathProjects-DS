import time
import numpy as np
from scipy.optimize import line_search


def gradient_descent(point,
                     lr,
                     gradient,
                     steps=1,
                     prj=lambda x, y: np.array([x, y]),
                     isTimed=False,
                     runFor=1):
    # starting point - point
    # lr - learning rate
    # gradient -  gradient function
    # steps -  how many times to perform the algorithm
    # prj - projection for PGD
    results = []
    start_time = time.time()
    step = 0
    while step < steps:
        if isTimed:
            step = -1
            elapsed = time.time() - start_time
        new_point = point - lr * gradient(point)
        results.append(prj(*new_point))
        point = new_point
        if (isTimed and elapsed >= runFor):
            print('ELAPSED:', elapsed)
            return np.array(results)
        step += 1
    return np.array(results)


def polyak(point,
           lr,
           beta,
           gradient,
           steps=1,
           prj=lambda x, y: np.array([x, y]),
           isTimed=False,
           runFor=1):
    previous = point
    results = []
    start_time = time.time()
    step = 0
    while step < steps:
        if isTimed:
            step = -1
            elapsed = time.time() - start_time
        new_point = point - lr * gradient(point) + beta * (point - previous)
        results.append(prj(*new_point))
        previous = point
        point = new_point
        if (isTimed and elapsed >= runFor):
            print('ELAPSED:', elapsed)
            return np.array(results)
        step += 1
    return np.array(results)


def nesterov(point,
             lr,
             beta,
             gradient,
             steps=1,
             prj=lambda x, y: np.array([x, y]),
             isTimed=False,
             runFor=1):
    previous = point
    results = []
    start_time = time.time()
    step = 0
    while step < steps:
        if isTimed:
            step = -1
            elapsed = time.time() - start_time
        new_point = point - lr * gradient(
            (point + beta * (point - previous))) + beta * (point - previous)
        results.append(prj(*new_point))
        previous = point
        point = new_point
        if (isTimed and elapsed >= runFor):
            print('ELAPSED:', elapsed)
            return np.array(results)
        step += 1
    return np.array(results)


def adagrad(point,
            lr,
            gradient,
            steps=1,
            prj=lambda arr: np.array(arr),
            isTimed=False,
            runFor=1):
    eps = 1e-8
    d = 0
    results = []
    start_time = time.time()
    step = 0
    while step < steps:
        if isTimed:
            step = -1
            elapsed = time.time() - start_time
        grad = gradient(point)
        d = d + grad**2
        new_point = point - (lr / np.sqrt(d + eps)) * grad
        results.append(prj(new_point))
        point = new_point
        if (isTimed and elapsed >= runFor):
            print('ELAPSED:', elapsed)
            return np.array(results)
        step += 1
    return np.array(results)


def newton(point, gradient1, gradient2, steps=1, isTimed=False, runFor=1):
    results = []
    start_time = time.time()
    step = 0
    while step < steps:
        if isTimed:
            step = -1
            elapsed = time.time() - start_time
        new_point = point - gradient1(point) / gradient2(point)
        point = new_point
        results.append(new_point)
        if (isTimed and elapsed >= runFor):
            print('ELAPSED:', elapsed)
            return np.array(results)
        step += 1
    return np.array(results)


def bfgs(point, func, gradient, steps=1, isTimed=False, runFor=1):
    results = []
    I = np.eye(point.size, dtype=int)
    H = I
    eps = 1e-8
    start_time = time.time()
    step = 0
    while step < steps:
        if isTimed:
            step = -1
            elapsed = time.time() - start_time
        grad = gradient(point)
        pk = -np.dot(H, grad)
        alpha = line_search(func, gradient, point, pk, maxiter=1000)[0]
        if (not alpha):
            print('Line Search of BFGS Failed at point:', point)
            return results
        new_point = point + alpha * pk
        results.append(new_point)
        if (np.linalg.norm(gradient(new_point)) < eps):
            return np.array(results)
        sk = new_point - point
        point = new_point
        new_grad = gradient(new_point)
        yk = new_grad - grad
        ro = 1.0 / (np.dot(yk, sk))
        A1 = I - ro * sk[:, np.newaxis] * yk[np.newaxis, :]
        A2 = I - ro * yk[:, np.newaxis] * sk[np.newaxis, :]
        H = np.dot(A1, np.dot(
            H, A2)) + (ro * sk[:, np.newaxis] * sk[np.newaxis, :])
        if (isTimed and elapsed >= runFor):
            print('ELAPSED:', elapsed)
            return np.array(results)
        step += 1
    return np.array(results)


STEPS = 20


def f_analysis(func,
               fprim,
               fprimprim,
               point,
               _min,
               st=STEPS,
               prj=lambda x, y, z: np.array([x, y, z])):
    points = gradient_descent(point, 0.001, fprim, st, prj=prj)
    print("gradient_descent:", abs(func(_min) - func(points[-1])))
    points = polyak(point, 0.001, 0.0009, fprim, st, prj=prj)
    print("Polyak:", abs(func(_min) - func(points[-1])))
    points = nesterov(point, 0.001, 0.0009, fprim, st, prj=prj)
    print("Nesterov:", abs(func(_min) - func(points[-1])))
    points = adagrad(point, 1.5, fprim, st)
    print("Adagrad:", abs(func(_min) - func(points[-1])))
    points = newton(point, fprim, fprimprim, st)
    print("Newton:", abs(func(_min) - func(points[-1])))
    points = bfgs(point, func, fprim, st)
    print("BFGS:", abs(func(_min) - func(points[-1])))


def f_analysis_time(func,
                    fprim,
                    fprimprim,
                    point,
                    _min,
                    _time,
                    prj=lambda x, y, z: np.array([x, y, z])):
    st = STEPS
    points = gradient_descent(point,
                              0.001,
                              fprim,
                              st,
                              prj=prj,
                              isTimed=True,
                              runFor=_time)
    print(f'Gradient_descent({len(points)} steps)',
          abs(func(_min) - func(points[-1])))
    points = polyak(point,
                    0.001,
                    0.0009,
                    fprim,
                    st,
                    prj=prj,
                    isTimed=True,
                    runFor=_time)
    print(f'Polyak({len(points)} steps)', abs(func(_min) - func(points[-1])))
    points = nesterov(point,
                      0.001,
                      0.0009,
                      fprim,
                      st,
                      prj=prj,
                      isTimed=True,
                      runFor=_time)
    print(f'Nesterov({len(points)} steps)', abs(func(_min) - func(points[-1])))
    points = adagrad(point, 1.5, fprim, st, isTimed=True, runFor=_time)
    print(f'Adagrad({len(points)} steps)', abs(func(_min) - func(points[-1])))
    points = newton(point, fprim, fprimprim, st, isTimed=True, runFor=_time)
    print(f'Newton({len(points)} steps)', abs(func(_min) - func(points[-1])))
    points = bfgs(point, func, fprim, st, isTimed=True, runFor=_time)
    print(f'BFGS({len(points)} steps)', abs(func(_min) - func(points[-1])))
