
# Importing the necessary packages, classes and modules.
import conv
import cv2, torch, numpy as np, time, matplotlib.pyplot as plt 


#========== PART A ==========#
in_image = cv2.imread("image3_1280x720.png")    # Reading the image using OpenCV.
#in_image = cv2.imread("image4_1920x1080.png")

#cv2.imshow('Input', in_image)    # Displaying the input image.
in_image_tensor = torch.from_numpy(in_image)    # Converting the input image, which is a numpy.ndarray into a torch tensor.

#========== Task 1 ==========#
# Format of Conv2D: Conv2D(in_channel, o_channel, kernel_size, stride, mode)
conv2d = conv.Conv2D(3,1,3,1,'known') # Defining the class object conv2d of class Conv2D

# The forward method of the Conv2D class will return the convolved output and also a count of the number of operations.
number_of_operations, out_image_tensor = conv2d.forward(in_image_tensor)    # Calling the forward method of the conv2d object with the input image tensor.
print('o_channels = {0} kernel_size = {1}, strides = {2}, number_of_operations = {3}'.format(1, 3, 1, number_of_operations)) # Printing the number of operations done.
conv2d.normalize_and_save(out_image_tensor) # Normalizing and saving the image.

#========== PART A ==========#
#========== Task 2 ==========#
# Format of Conv2D: Conv2D(in_channel, o_channel, kernel_size, stride, mode)
conv2d = conv.Conv2D(3,2,5,1,'known') # Defining the class object conv2d of class Conv2D

# The forward method of the Conv2D class will return the convolved output and also a count of the number of operations.
number_of_operations, out_image_tensor = conv2d.forward(in_image_tensor)    # Calling the forward method of the conv2d object with the input image tensor.
print('o_channels = {0} kernel_size = {1}, strides = {2}, number_of_operations = {3}'.format(2, 5, 1, number_of_operations)) # Printing the number of operations done.
conv2d.normalize_and_save(out_image_tensor) # Normalizing and saving the image.

#========== PART A ==========#
#========== Task 3 ==========#
# Format of Conv2D: Conv2D(in_channel, o_channel, kernel_size, stride, mode)
conv2d = conv.Conv2D(3,3,3,2,'known') # Defining the class object conv2d of class Conv2D

# The forward method of the Conv2D class will return the convolved output and also a count of the number of operations.
number_of_operations, out_image_tensor = conv2d.forward(in_image_tensor)    # Calling the forward method of the conv2d object with the input image tensor.
print('o_channels = {0} kernel_size = {1}, strides = {2}, number_of_operations = {3}'.format(3, 3, 2, number_of_operations)) # Printing the number of operations done.
conv2d.normalize_and_save(out_image_tensor) # Normalizing and saving the image.


#========== PART B ==========#
in_image = cv2.imread("image3_1280x720.png")    # Reading the image using OpenCV.
## Assumption:   stride = 1, kernel_size = 3 and size of image is 1280 x 720 pixels (colored).
print('\nAssumption:   stride = 1, kernel_size = 3 and size of image is 1280 x 720 pixels (colored).\n')

in_image = cv2.imread("image2_1920x1080.png")    # Reading the image using OpenCV.
### Assumption:   stride = 1, kernel_size = 3 and size of image is 1920 x 1080 pixels (colored).
print('\nAssumption:   stride = 1, kernel_size = 3 and size of image is 1920 x 1080 pixels (colored).\n')

cv2.imshow('Input', in_image)    # Displaying the input image.
in_image_tensor = torch.from_numpy(in_image)    # Converting the input image, which is a numpy.ndarray into a torch tensor.

time_taken = [] # This list will hold the value of the amount of time taken for the convolution with different i, where 2**i = number of o_channels.
idx = []    # This is to collect the values of i.

for i in range(11):
    start_time = time.time()  # Recording the start time.
    
    ## Format of Conv2D: Conv2D(in_channel, o_channel, kernel_size, stride, mode)
    conv2d = conv.Conv2D(3,2**i,3,1,'rand') # Defining the class object conv2d of class Conv2D
    ## The forward method of the Conv2D class will return the convolved output and also a count of the number of operations.
    number_of_operations, out_image_tensor = conv2d.forward(in_image_tensor)    # Calling the forward method of the conv2d object with the input image tensor.

    end_time = time.time()  # Recording the end time.
    
    time_taken.append(end_time - start_time)    # Calculating the time taken in seconds, and storing it in a list.
    idx.append(i)   # These are the corresponding i values.
    
    print('i = {0}, o_channel = {1}, number_of_operations = {2}, time_taken = {3} sec'.format(i, 2**i, number_of_operations, time_taken[i])) # Printing the necessary information.
    conv2d.normalize_and_save(out_image_tensor) # Normalizing and saving the image.
    
## Plotting the variations of the time taken for different values of i.
plt.plot(idx, time_taken, 'rs', idx, time_taken, '--k', linewidth = 2)  # Plotting the curve with red squares and black dotted lines.
plt.grid(True)  # Display grid.
plt.title('Time variations with the change of i')   # Plot title.
plt.xlabel('i') # X label.
plt.ylabel('Time (seconds)')    # Y label.
plt.show()   # Showing the plot.


##========== PART C ==========#
in_image = cv2.imread("image3_1280x720.png")    # Reading the image using OpenCV.
## Assumption:   stride = 1, kernel_size = 3 and size of image is 1280 x 720 pixels (colored).
print('\nAssumption:   stride = 1, o_channel = 2 and size of image is 1280 x 720 pixels (colored).\n')

in_image = cv2.imread("image2_1920x1080.png")    # Reading the image using OpenCV.
### Assumption:   stride = 1, kernel_size = 3 and size of image is 1920 x 1080 pixels (colored).
print('\nAssumption:   stride = 1, o_channel = 2 and size of image is 1920 x 1080 pixels (colored).\n')

cv2.imshow('Input', in_image)    # Displaying the input image.
in_image_tensor = torch.from_numpy(in_image)    # Converting the input image, which is a numpy.ndarray into a torch tensor.

operations_done = [] # This list will hold the value of the amount of time taken for the convolution with different i, where 2**i = number of o_channels.
idx = []    # This is to collect the values of i.

for i in range(3, 12, 2):
    ## Format of Conv2D: Conv2D(in_channel, o_channel, kernel_size, stride, mode)
    conv2d = conv.Conv2D(3,2,i,1,'rand') # Defining the class object conv2d of class Conv2D
    ## The forward method of the Conv2D class will return the convolved output and also a count of the number of operations.
    number_of_operations, out_image_tensor = conv2d.forward(in_image_tensor)    # Calling the forward method of the conv2d object with the input image tensor.

    operations_done.append(number_of_operations)    # Calculating the number of operations done, and storing it in a list.
    idx.append(i)   # These are the corresponding i values.
    
    print('kernel_size = {0}, number_of_operations = {1}'.format(i, number_of_operations)) # Printing the necessary information.
    conv2d.normalize_and_save(out_image_tensor) # Normalizing and saving the image.
    
## Plotting the variations of the time taken for different values of i.
plt.plot(idx, operations_done, 'go', idx, operations_done, '--k', linewidth = 2)    # Plotting the curve with red squares and black dotted lines.
plt.grid(True)  # Display grid.
plt.title('Number of operations variations vs Kernel size')    # Plot title.
plt.xlabel('Kernel size') # X label.
plt.ylabel('Number of operations')    # Y label.
plt.show()   # Showing the plot.

