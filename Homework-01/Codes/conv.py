
# Importing the necessary packages and modules.
import numpy as np, torch, cv2

class Conv2D(object):

    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        
        # Defining the Kernels.
        self.K1 = torch.Tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        self.K2 = torch.Tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        self.K3 = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.K4 = torch.Tensor([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        self.K5 = torch.Tensor([[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]])

        # Selecting the Kernel based on the input arguments.
        if self.in_channel == 3 and self.o_channel == 1 and mode == 'known':
            self.list_of_kernels = [self.K1]  
        elif self.in_channel == 3 and self.o_channel == 2 and mode == 'known':
            self.list_of_kernels = [self.K4, self.K5]  
        elif self.in_channel == 3 and self.o_channel == 3 and mode == 'known':
            self.list_of_kernels = [self.K1, self.K2, self.K3]  
        elif self.in_channel == 3 and mode == 'rand':  
            self.list_of_kernels = []   
            for i in range(self.o_channel):
                k = torch.randn(self.kernel_size, self.kernel_size) 
                self.list_of_kernels.append(k)  

            #self.fileObj = open('kernel_file_Part_B.txt','w') # File to note down the kernels in case of 'rand' mode.
            #self.fileObj.write('Assumption:     stride = 1, kernel_size = 3, size of the image = 1920 x 1080 (colored).\n\n')
        else:
            print('\n\nIncorrect input parameters to the Class Conv2D\n\n')  # Error message.
    

    def forward(self, input_tensor):
        # Convolution.
        input_tensor = input_tensor.type(torch.FloatTensor) # Converting the input tensor into FloatTensor type.
        #print(input_tensor.size(), type(input_tensor))

        rows = input_tensor.size(0)    # Measuring the height of the input tensor.
        cols = input_tensor.size(1)    # Measuring the height of the input tensor.

        sum_of_all_channels = torch.zeros(rows, cols)  # Creating an empty tensor which will later hold the sum of all corresponding elements of all the channels. 
        # This should have only one channel since it will only hold the sum. But the size of the channel should be same as that of the image.
        
        if self.in_channel > 1: # Checking if the number of channels is greater than 1 or not.
            for i in range(0, self.in_channel):
                sum_of_all_channels += input_tensor[:,:,i]   # Adding all the channels of the input tensor, if it has multiple channels (like red, green, blue etc.).
        else:
            sum_of_all_channels = input_tensor   # In case the input tensor has only one channel, no addition is needed.
        
        # The convolution will occur between a matrix called input_matrix and another called kernel_matrix. The kernel_matrix is assumed to be the smaller of the two.
        input_matrix = sum_of_all_channels  # This will be the matrix to be convolved with the kernel matrix.

        list_of_output_tensors = []    # Creating a blank list, that will hold the output tensors.
        
        # Scanning through all the kernels inside the list of kernels and convolving with each of them one by one, and storing the outputs into the list of output tensors.
        for i in range(0, len(self.list_of_kernels)):
            kernel_matrix = self.list_of_kernels[i];   # Selecting the kernels from the list of kernels, one by one.
            
            #if self.mode == 'rand':
                #self.fileObj.write('kernel_{0}{1}\n'.format((i+1), kernel_matrix))
                ##print('kernel_{0}\n{1}'.format((i+1), kernel_matrix)) # Printing the kernel in case of 'rand' mode.
                
            kernel_rows = kernel_matrix.size(0) # No. of kernel rows.
            row_start_idx = kernel_rows//2 # This is the start row index for the convolution, there will be a border of zeros all around the output matrix.
            row_end_idx = rows - kernel_rows//2 -1 # This is the last row index for the convolution.

            kernel_cols = kernel_matrix.size(1) # No. of kernel cols.
            col_start_idx = kernel_cols//2 # This is the start col index for the convolution.
            col_end_idx = cols - kernel_cols//2 -1 # This is the last row index for the convolution.

            output_tensor = torch.zeros(rows//self.stride, cols//self.stride)    # Creating an empty tensor that will hold the result of convolution. If the stride is > 1, then we will have a reduced image.
            #print self.sum_of_all_channels.size()

            for r in range(row_start_idx, row_end_idx + 1, self.stride):  # Scanning through all the rows. The +1 is because python excludes the last index.
                for c in range(col_start_idx, col_end_idx + 1, self.stride):    # Scanning through all the cols. The +1 is because python excludes the last index.
                    
                    # Cropping out a kernel sized submatrix from the input matrix. The +1 is because python excludes the last index.
                    input_submatrix = input_matrix[ r-kernel_rows//2 : r+kernel_rows//2 + 1 , c-kernel_cols//2 : c+kernel_cols//2 + 1 ]
                    submatrix_and_kernel_product = input_submatrix * kernel_matrix   # Mutiplying.
                    output_tensor[ r//self.stride, c//self.stride ] = submatrix_and_kernel_product.sum()   # Adding.
                    # The mapping is like this if the stride is two, 1,3,5,7... -> 0,1,2,3... For stride equal to three, the mapping is 1,4,7,10... -> 0,1,2,3...
                    
            list_of_output_tensors.append(output_tensor)  # Storing the tensor into the list of output tensors.

        complete_output_tensor = torch.stack(list_of_output_tensors, 2)   # Converting the entire list of output tensors into a fresh output tensor.
        #print(type(output_tensor), output_tensor.size(), type(complete_output_tensor), complete_output_tensor.size())
                    
        # Calculating the number of additions and multiplications performed by the convolution. First term is the no. of additions when the channels of the input are added together.
        # Second term is the number of multiplications and the number of additions performed when the convolution is done, (kernel_rows*kernel_cols = number of multiplications and 
        # kernel_rows*kernel_cols -1 is the number of additions for each step of the inner for loop.
        # This second term will be repeated for every kernel in the list of kernels. So the o_channel is multiplied to it.
        number_of_operations = rows*cols*(self.in_channel-1) + (row_end_idx)*(col_end_idx)*((kernel_rows)*(kernel_cols)*2 - 1)*self.o_channel

        #self.fileObj.close();   # Closing the file.

        return(number_of_operations, complete_output_tensor)  # Returning the number of calculation count and the convolution result.


    def normalize_and_save(self, input_tensor):
        # Saving each of the channels of the input tensor as a normalized image.
        for i in range(0, input_tensor.size(2)):
            # Normalizing the output tensor to the range of 0 to 255, so that it can be shown as an embossed image.
            # If you are trying to view the image directly from array from inside the program, then you have to use 1.0 instead of 255.0.
            # But if you are saving the image and then reading it with imread, then you have to use 255.0.
            out_image_tensor_normalized = (input_tensor[:,:,i] - torch.min(input_tensor[:,:,i])) * 255.0 / (torch.max(input_tensor[:,:,i]) - torch.min(input_tensor[:,:,i]))

            out_image = out_image_tensor_normalized.numpy()   # Now converting the output tensor into a numpy array for displaying using OpenCV.
            filename = 'output_' + str(input_tensor.size(1)) + 'x' + str(input_tensor.size(0)) + '_' + str(self.in_channel) + str(self.o_channel) + str(self.kernel_size) + str(self.stride) \
                       + '_' + str(i+1) + '.png'   # Creating the filename for the saved file.
            cv2.imwrite(filename, out_image)   # Saving the image as a png file.


        
        
        
