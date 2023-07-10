import matplotlib.pyplot as plt

def drawmtx(x):
    plt.imshow(x.numpy(), cmap='gray')
    plt.colorbar()
    plt.show()

def show():
    plt.show()