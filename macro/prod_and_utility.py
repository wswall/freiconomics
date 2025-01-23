import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# Path to the folder where you want to save the graphs
OUTPUT_DIRECTORY = "../figures"


def cobb_douglas(labor, alpha):
    return labor ** (1 - alpha)


def calculate_output(labor, alpha, theta):
    production = cobb_douglas(labor, alpha)
    return theta * production


def log_utility(c, sigma):
    if sigma == 1:
        return np.log(c)
    return (c ** (1 - sigma) - 1) / (1 - sigma)


def disutility_from_labor(labor, delta, phi):
    return -delta / (1 + phi) * labor ** (1 + phi)


if __name__ == "__main__":

    # Generate array of 10000 evenly spaced numbers in the range 0 to 2.5, inclusive
    l_space = np.linspace(0, 2.5, 10000)

    # Generate plots of output, varying alpha and holding theta constant at 10
    for a in [0.25, 0.5, 0.75]:

        # Calculate output for each element in l_space for current alpha
        output = calculate_output(l_space, a, 10)

        # Plot the output, using latex to label with an alpha symbol and
        # inserting the current value of alpha into the label
        plt.plot(l_space, output, label=fr"$\alpha={a}$")

    # Add labels on the x and y axis for labor at time t and output at
    # time t, using latex to render with t subscript
    plt.xlabel("$l_t$")
    plt.ylabel("$y_t$")

    # Calling .legend() ensures the lines' labels are displayed
    plt.legend()


    # Generate plots of output, varying theta and holding alpha constant at 0.5
    for t in [5, 10, 15]:
        output = calculate_output(l_space, 0.5, t)
        plt.plot(l_space, output, label=fr"$\theta={t}$")
    plt.xlabel("$l_t$")
    plt.ylabel("$y_t$")
    plt.legend()

    # Generate plots of disutility of labor, varying phi and holding delta constant
    for p in [0.5, 1, 2]:
        disutility = disutility_from_labor(l_space, 2, p)
        plt.plot(l_space, disutility, label=fr"$\phi={p}$")
    plt.xlabel("$l_t$")
    plt.ylabel("$f(l_t)$")
    plt.legend()

    # Generate array of 10000 evenly spaced numbers in the range 0.5 to 2, inclusive
    c_space = np.linspace(0.5, 2, 10000)
    for s in [0.5, 1, 2]:
        utility = log_utility(c_space, s)
        plt.plot(c_space, utility, label=fr"$\sigma={s}$")
    plt.xlabel("$c_t$")
    plt.ylabel("$v(c_t)$")
    plt.legend()
