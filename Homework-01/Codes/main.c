#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

// Functions.
double c_conv(int in_channel, long int o_channel, int kernel_size, int stride, float***, int rows, int cols);


int main (int argc, char** argv)
{
    // Creating variables.
    int rows = 720, cols = 1280;
//     int rows = 1080, cols = 1920;

    float ***input_array, time_taken;
    double number_of_operations = 0.0;
    int in_channel = 3, kernel_size = 3, stride = 1;
    int n, r, c, i;
    long int o_channel = 1;
    clock_t start_time, end_time;
    
    input_array = (float***)malloc(sizeof(float**)*in_channel);
    for(n = 0; n < in_channel; n++)
        input_array[n] = (float**)malloc(sizeof(float*)*rows);
    
    for(n = 0; n < in_channel; n++)
        for(r = 0; r < rows; r++)
            input_array[n][r] = (float*)malloc(sizeof(float)*cols);

    // Creating input array with random numbers.
    for(n = 0; n < in_channel; n++)
        for(r = 0; r < rows; r++)
            for(c = 0; c < cols; c++)
                input_array[n][r][c] = (float)(rand()%255);
    
    printf("\n Assumption:   stride = 1, kernel_size = 3 and size of image is %d x %d pixels (colored)\n", cols, rows);

    for(i = 0; i < 11; i++)
    {
        start_time = clock();
        number_of_operations = c_conv(in_channel, pow(2,i), kernel_size, stride, input_array, rows, cols);
        end_time = clock();
        
        time_taken = ((float)(end_time - start_time))/CLOCKS_PER_SEC;
        printf("i = %d, o_channel = %f, number_of_operations = %10.0lf, time taken = %f seconds \n", i, pow(2,i), number_of_operations, time_taken);
    }// for.
    
    printf("\n");

    free(input_array);

    return 0;

}// main.



double c_conv(int in_channel, long int o_channel, int kernel_size, int stride, float*** input_array, int rows, int cols)
{
    // Creating kernel.
    float **kernel, **output_array, **sum_of_all_channels;
    double number_of_operations = 0.0;
    int n, r, c, i, rk, ck;
    int kernel_rows = kernel_size, row_start_idx, row_end_idx, kernel_cols = kernel_size, col_start_idx, col_end_idx;
    
    // Creating the kernel.
    kernel = (float**)malloc(sizeof(float*)*kernel_rows);
    for(r = 0; r < kernel_rows; r++)
        kernel[r] = (float*)malloc(sizeof(float)*kernel_cols);

    // Creating an array that will hold the sum of all the corresponding elements of the input array and also creating an output array of same size.
    sum_of_all_channels = (float**)malloc(sizeof(float*)*rows);
    output_array = (float**)malloc(sizeof(float*)*rows);
    for(r = 0; r < rows; r++)
    {
        sum_of_all_channels[r] = (float*)malloc(sizeof(float)*cols);
        output_array[r] = (float*)malloc(sizeof(float)*cols);
    }// for.
    
    // Initializing the output array and also calculating the sum_of_all_channels array elements values.
    for(r = 0; r < rows; r++)
        for(c = 0; c < cols; c++)
        {
            output_array[r][c] = 0.0;
            sum_of_all_channels[r][c] = input_array[0][r][c] + input_array[1][r][c] + input_array[2][r][c];
            number_of_operations = number_of_operations + 2.0;
        }// for.

    for(i = 0; i < o_channel; i++)
    {
        // Initializing the kernel to random values.
        for(r = 0; r < kernel_size; r++)
            for(c = 0; c < kernel_size; c++)
                kernel[r][c] = ((float)(rand()%10000) - 5000.0)/5000.0;
            
        row_start_idx = kernel_rows/2;  // This is the start row index for the convolution, there will be a border of zeros all around the output matrix.
        row_end_idx = rows - kernel_rows/2 -1;  // This is the last row index for the convolution.

        col_start_idx = kernel_cols/2;  // This is the start col index for the convolution.
        col_end_idx = cols - kernel_cols/2 -1;  // This is the last row index for the convolution.

        // Convolution.
        for(r = row_start_idx; r < row_end_idx +1; r += stride)
            for(c = col_start_idx; c < col_end_idx +1; c += stride)
            {   for(rk = -1*kernel_rows/2; rk < kernel_rows/2 +1; rk++)
                    for(ck = -1*kernel_cols/2; ck < kernel_cols/2 +1; ck++)
                    {    output_array[r][c] += kernel[ kernel_rows/2 + rk ][ kernel_cols/2 + ck ] * sum_of_all_channels[ r + rk ][ c + ck ];
                        // The mapping is like this if the stride is two, 1,3,5,7... -> 0,1,2,3... For stride equal to three, the mapping is 1,4,7,10... -> 0,1,2,3...
                        number_of_operations = number_of_operations + 2;
                    }// for.
                
//                 number_of_operations = number_of_operations - 1;
            }// for.
                    
//         number_of_operations);
    }// for.

    free(output_array);
    free(sum_of_all_channels);
    free(kernel);
    
    return(number_of_operations);
    
}// c_conv.



