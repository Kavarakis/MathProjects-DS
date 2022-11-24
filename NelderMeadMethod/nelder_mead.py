import numpy as np
import copy
import time
from scipy.spatial.distance import cdist

EPS = 1e-10


def edist(
    a,
    b,
):
    arr = np.zeros(shape=a.shape[0])
    for i in range(a.shape[0]):

        res = np.sqrt(np.sum((a[i, :] - b[i, :])**2))
        arr[i] = res

    return arr


def nm(f, x):
    step_count = 0
    start = time.time()
    passed_01, passed_1, passed_2 = False, False, False
    while True:
        xnew = copy.deepcopy(x)
        y_val = np.apply_along_axis(f, 1, x)
        y_arg = np.argsort(y_val)
        worst = x[y_arg[-1]]
        remaining_points = x[y_arg[:-1]]
        centroid = np.mean(remaining_points, axis=0)
        refl_point = worst + 2 * (centroid - worst)
        refl_point_eval = f(refl_point)
        if refl_point_eval < y_val[y_arg[0]]:  ## Better than previous best
            exp_point = worst + 3 * (centroid - worst)
            if f(exp_point) > refl_point_eval:
                xnew[y_arg[-1]] = exp_point  ## EXPANSION
                #print("EXPANSION")
            else:
                xnew[y_arg[-1]] = refl_point  ## REFLECTION
                #print("REFLECTION")
        elif refl_point_eval < y_val[y_arg[-2]]:  ## Better than lousy point
            xnew[y_arg[-1]] = refl_point
            #print("REFLECTION")
        elif refl_point_eval < y_val[y_arg[-1]]:  ## Better than worst point
            contr_point = worst + 3 * (centroid - worst) / 2
            if f(contr_point) < refl_point_eval:
                xnew[y_arg[-1]] = contr_point  ## OUTSIDE CONTRACTION
                #print("OUTSIDE CONTRACTION")
            else:
                best = x[y_arg[0]]
                xnew = np.array([(best + xi) / 2 for xi in x])  ## SHRINK
                #print("SHRINK")
        else:  ## Worse than worst point
            contr_point = worst + (centroid - worst) / 2
            if f(contr_point) < f(worst):
                xnew[y_arg[-1]] = contr_point  ## INSIDE CONTRACTION
                #print("INSIDE CONTRACTION")
            else:
                best = x[y_arg[0]]
                xnew = np.array([(best + xi) / 2 for xi in x])  ## SHRINK
                #print("SHRINK")

        distances = edist(xnew, x)
        # if np.amax(distances) < EPS:
        #     break
        x_temp = x
        x = xnew
        step_count = step_count + 1
        x_min = x[np.argsort(np.apply_along_axis(f, 1, x_temp))[0], :]
        if step_count == 2:
            print('Passed 2 iterations, current best point: ', np.min(y_val),
                  x_min, f(x_min))
        elif step_count == 5:
            print('Passed 5 iterations, current best point: ', np.min(y_val),
                  x_min, f(x_min))
        elif step_count == 10:
            print('Passed 10 iterations, current best point: ', np.min(y_val),
                  x_min, f(x_min))
        elif step_count == 50:
            print('Passed 50 iterations, current best point: ', np.min(y_val),
                  x_min, f(x_min))
        elif step_count == 100:
            print('Passed 100 iterations, current best point: ', np.min(y_val),
                  x_min, f(x_min))

        # print(
        #     f'{step_count}. iteration, current best point: {np.min(y_val)}, max distance between points: {np.amax(distances)}'
        # )
        if time.time() - start > 0.1 and not passed_01:
            print(
                f'In 0.1 seconds completed {step_count} iterations {x_min},{f(x_min)}'
            )
            passed_01 = True
        if time.time() - start > 1 and not passed_1:
            print(
                f'In 1 seconds completed {step_count} iterations {x_min},{f(x_min)}'
            )
            passed_1 = True
        if time.time() - start > 2 and not passed_2:
            print(
                f'In 2 seconds completed {step_count} iterations {x_min},{f(x_min)}'
            )
            passed_2 = True
            break
    print(
        f'{step_count}. iteration, current best point: {np.min(y_val)}, max distance between points: {np.amax(distances)}, point {x_min}, value { f(x_min)}'
    )
    ymin = np.apply_along_axis(f, 1, xnew)
    y_arg_min = np.argsort(ymin)
    return xnew[y_arg_min[0]], f(xnew[y_arg_min[0]])
