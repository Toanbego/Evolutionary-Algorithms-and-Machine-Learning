import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', '-m', type=str,
                        help='Pick either gradient_search or exhaustive_search')

    # For gradient
    parser.add_argument('--gradient_method', '-g', type=str, help='Pick either ascent or descent', default='ascent')
    parser.add_argument('--exploration', '-e', type=int, help='Number of explorations', default=0)
    parser.add_argument('--start_point', '-s', type=float, help='Start point of gradient search', default=3)
    return parser.parse_args()


def plot(f = 1 , x = [0,1,2,3], scatter = False):
    """
    Adds the results to the plot. does not show the plot
    Use plt.show() in main function for that
    :param f: function to plot
    :param x: x interval or point to plot
    :param scatter: True when plotting local max/min
    :return:
    """

    if scatter == True:
        plt.scatter(x, f(x))
    elif scatter == False:
        plt.plot(x, f(x))


    # Config the graph
    plt.title('Optimize methods')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend(['f(x) = -x^4 + 2x^3 + 2x^2 - x',
                "f'(x) = -4x^3 + 6x^2 + 4x - 1"], loc='upper left')



def gradient_search(f_derv, start_point=3, descent_or_ascent="ascent"):

    """
    Function performs gradient ascent/descent for
    given function derivative with a start point.
    Returns the local maximum
    :param f_derv: Derivative of function
    :param start_point: Starting point for search
    :return: local maximum
    """

    # Initialize parameters
    cur_x = start_point  # The algorithm starts at x=3
    rate = 0.01  # Learning rate
    precision = 0.000001  # This tells us when to stop the algorithm
    previous_step_size = 1  #
    max_iters = 10000  # maximum number of iterations
    iters = 0  # iteration counter

    # Gradient ascent with extra exploration



    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x  # Store current x value in prev_x
        if descent_or_ascent == "descent":
            cur_x = cur_x - rate * f_derv(prev_x)  # Grad descent
        if descent_or_ascent == "ascent":
            cur_x = cur_x + rate * f_derv(prev_x)  # Grad descent
        previous_step_size = abs(cur_x - prev_x)  # Change in x
        iters = iters + 1  # iteration count


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
    # Parse arguments
    args = parse_arguments()

    # Functions and interval
    f = lambda x: -x**4 + 2*x**3 + 2*x**2 - x
    f_derv = lambda x: -4 * x ** 3 + 6 * x ** 2 + 4 * x - 1
    x = np.linspace(-2, 3)

    # Plot the function and its derivative
    plot(f, x)
    plot(f_derv, x)

    # Pick method to use
    if args.method == 'exhaustive_search':
        local_max_min = exhaustive_search(f)

    elif args.method == 'gradient_search':

        # Use exploration
        if args.exploration > 0:
            local_max_min = f(-2)
            for step in x[::int(len(x) / args.exploration)]:
                print("Exploring from: {}".format(step))
                new_value = gradient_search(f_derv, start_point=step,
                                            descent_or_ascent="ascent")
                if f(new_value) > f(local_max_min):
                    local_max_min = new_value

        # Do not use exploration, but chosen start point
        else:
            local_max_min = gradient_search(f_derv, args.start_point, args.gradient_method)
    # Raise an error if not valid method is chosen
    else:
        raise ValueError("Pick either gradient_search or exhaustive_search")

    # Plot the local max/min with the function
    plot(f, local_max_min, scatter=True)
    plt.show()




main()