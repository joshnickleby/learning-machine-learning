import numpy as np
import matplotlib.pyplot as plt


DEFAULT_LAMBDA = lambda item: item;

def getMinMaxByPosition(set, position=0, wrapperFn=DEFAULT_LAMBDA):
    subset = set[:, position]
    return wrapperFn(subset.min() - 1.0), wrapperFn(subset.max() + 1.0)



def getMinMaxFromSet(set, wrapperFn=DEFAULT_LAMBDA):
    xMin, xMax = getMinMaxByPosition(set, wrapperFn=wrapperFn)
    yMin, yMax = getMinMaxByPosition(set, 1, wrapperFn=wrapperFn)
    return xMin, xMax, yMin, yMax




# todo: WTF IS THIS DOING - FIND OUT - comments are not my own
def visualizeClassifier(classifier, x, y, title=''):
    # Define the minimum and maximum values for x and y that will be used in the mesh grid
    xMin, xMax, yMin, yMax = getMinMaxFromSet(x)

    # Define the step size to use in plotting the mesh grid
    meshStepSize = 0.01

    # Define the mesh grid of x and Y values
    xValues, yValues = np.meshgrid(np.arange(xMin, xMax, meshStepSize), np.arange(yMin, yMax, meshStepSize))

    # Run the classifier on the mesh grid
    output = classifier.predict(np.c_[xValues.ravel(), yValues.ravel()])

    # Reshape the output array
    output = output.reshape(xValues.shape)

    # Create a plot
    plt.figure()

    # Specify the title
    plt.title(title)

    # Choose a color scheme for the plot
    plt.pcolormesh(xValues, yValues, output, cmap=plt.cm.gray)

    # Overlay the training points on the plot
    plt.scatter(x[:, 0], x[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # Specify the boundaries of the plot
    plt.xlim(xValues.min(), xValues.max())
    plt.ylim(yValues.min(), yValues.max())

    makeInt = lambda item : int(item)

    xTickMin, xTickMax, yTickMin, yTickMax = getMinMaxFromSet(x, wrapperFn=makeInt)

    # Specify the ticks on the x and y axes
    plt.xticks((np.arange(xTickMin, xTickMax, 1.0)))
    plt.yticks((np.arange(yTickMin, yTickMax, 1.0)))

    plt.show()
