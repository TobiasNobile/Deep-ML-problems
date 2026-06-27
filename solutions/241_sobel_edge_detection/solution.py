import numpy as np

def sobel_edge_detection(image):
    """
    Apply Sobel edge detection to a grayscale image.
    
    Args:
        image: 2D list/array representing a grayscale image
               with values in range [0, 255]
    
    Returns:
        Edge magnitude image as 2D list with integer values (0-255),
        or -1 if input is invalid
    """
    image = np.array(image)
    img_in_range = ((0 <= image) &( image <=255)).all()
    if len(image.shape) != 2 or image.shape[0]< 3 or image.shape[1]<3 or not img_in_range:
        return -1

    Gx = np.array([
        [-1, 0, 1], 
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    Gy = np.array([
        [-1, -2, -1], 
        [0, 0, 0],
        [1, 2, 1]
    ])

    g_x, g_y = [], []
    for i in range(len(image)-2):
        row_x, row_y = [], []
        for j in range(image.shape[1]-2):
            window = image[i:i+3, j:j+3]
            product_x = np.multiply(window, Gx)
            product_y = np.multiply(window, Gy)

            sum_x = np.sum(product_x)
            sum_y = np.sum(product_y)
            row_x.append(sum_x)
            row_y.append(sum_y)
        
        g_x.append(row_x)
        g_y.append(row_y)

    g_x, g_y = np.array(g_x), np.array(g_y)
    g = np.sqrt(g_x**2 + g_y**2)
    if np.max(g) == 0:
        return [[0]]
    return (g/np.max(g)*255).tolist()

    
    