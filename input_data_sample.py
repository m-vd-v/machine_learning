import random

## Function may take an int (sample_index) as an argument, and will display the sample with that index.
## Otherwise, it will display a random sample. If a given sample_index is too high or low, it will
## display the sample with either the highest or lowest possible index respectively.
def input_data_sample(sample_index: int = None):
    if sample_index is None:
        sample_index = random.randrange(0, 1439)
    elif sample_index > 1439:
        sample_index = 1439
    elif sample_index < 0:
        sample_index = 0

    image_index = x_coords[sample_index, :]
    image = np.reshape(image_index, (-1, 32))

    plt.imshow(image, origin='lower')
    plt.title("Input data sample as an image")
    plt.show()