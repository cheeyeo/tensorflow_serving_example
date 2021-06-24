import matplotlib.pyplot as plt

def show(test_images, idx, title):
    plt.figure()
    img = test_images[idx].reshape(28, 28)
    print(img.shape, img.dtype)
    plt.axis("off")
    plt.title(title)
    plt.imshow(img)
    plt.show()