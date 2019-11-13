from matplotlib import pyplot as plt

# Function for testing
# Maps x => x*x
def calculate(x):
	return x * x


inputs = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]

outputs = [calculate(x) for x in inputs]

plt.plot(inputs, outputs)
plt.savefig("plot.png")