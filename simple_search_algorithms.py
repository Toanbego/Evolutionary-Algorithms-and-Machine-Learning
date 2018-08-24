import numpy as np
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt


def plot(f = 1 , x = [0,1,2,3], scatter = False):


    if scatter == True:
        plt.scatter(exhaustive_search(f), f(exhaustive_search(f)))
    elif scatter == False:
        plt.plot(f, x)


    ## Config the graph
    plt.title('A Cool Graph')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.ylim([0, 50])
    plt.grid(True)
    plt.legend(['f(x) = -x^4 + 2x^3 + 2x^2 - x',
                "f'(x) = -4x^3 + 6x^2 + 4x - 1"], loc='upper left')



def gradient_ascent(f_derv, start_point = 3):
    """
    Function performs gradient ascent/descent for
    given function derivative with a
    start point. Returns the local maximum
    :param f_derv: Derivative of function
    :param start_point: Starting point for search
    :return: local maximum
    """
    cur_x = start_point  # The algorithm starts at x=3
    rate = 0.01  # Learning rate
    precision = 0.000001  # This tells us when to stop the algorithm
    previous_step_size = 1  #
    max_iters = 10000  # maximum number of iterations
    iters = 0  # iteration counter


    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x  # Store current x value in prev_x
        cur_x = cur_x + rate * f_derv(prev_x)  # Grad ascent
        previous_step_size = abs(cur_x - prev_x)  # Change in x
        iters = iters + 1  # iteration count
        print("Iteration", iters, "\nX value is", cur_x)  # Print iterations

    print("The local minimum occurs at", cur_x)
    return cur_x


def exhaustive_search(f):
    """
    Function that searches every possible solution and returns the best one
    :param f: Function
    :return: Returns y and x value
    """
    step = 0.5
    x = np.arange(-2, 3+step, step)
    max_value = f(3)
    for step in x:
        new_value = f(step)
        if new_value > max_value:
            max_value = new_value
            x_value = step

    return x_value


def main():

    #Function
    f = lambda x: -x**4 + 2*x**3 + 2*x**2 - x

    # Parameters for gradient decent
    x = np.linspace(-2, 3)
    f_derv = lambda x: -4*x**3 + 6*x**2 + 4*x - 1

   #TODO fix plotting function
    max_grad = gradient_ascent(f_derv)
    plot(f, x)

    max_exhaust = exhaustive_search(f)
    plot(f, x)


    plt.show()




main()