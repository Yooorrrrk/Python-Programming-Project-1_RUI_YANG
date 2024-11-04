import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime
import random
from sklearn.cluster import KMeans


def read_image(path, show=False):
    '''
    Return a Numpy array which contains the information of readed image;
    
    If the parameter [show] is True (which default is False):
    The result will also show a row of image (1 x 4)
    In sequence are the color image, and the images for the red, green, and blue channels.
    '''
    
    ## Read image from path
    image_array = plt.imread(path)
    
    ## If [show] == True, plot 4 images.
    if show:
        ## Replicate the complete 3-channel matrix four times.
        Color_image = image_array.copy()
        Red_layer = image_array.copy()
        Green_layer = image_array.copy()
        Blue_layer = image_array.copy()

        ## For one specific layer, we need to close another two channel.
        Red_layer[:, :, 1] = 0      # close green channel
        Red_layer[:, :, 2] = 0      # close blue channel

        Green_layer[:, :, 0] = 0    # close red channel
        Green_layer[:, :, 2] = 0    # close blue channel

        Blue_layer[:, :, 0] = 0     # close red channel
        Blue_layer[:, :, 1] = 0     # close green channel

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        ## Plot color image
        axes[0].imshow(Color_image)
        axes[0].set_title("Color Image")
        axes[0].axis('off')

        ## Plot red channel image
        axes[1].imshow(Red_layer)
        axes[1].set_title("Image of Red Channel")
        axes[1].axis('off')

        ## Plot green channel image
        axes[2].imshow(Green_layer)
        axes[2].set_title("Image of Green Channel")
        axes[2].axis('off')
        
        ## Plot blue channel image
        axes[3].imshow(Blue_layer)
        axes[3].set_title("Image of Blue Channel")
        axes[3].axis('off')

        plt.show()
        
    ## If [show] == False, do nothing.
    else:
        pass
    
    return image_array

def get_clock_hands(clock_RGB):
    '''
    Return the exact coordinates of hour hand and minute hand.
    
    Step 1: find the coordinates of grey pixels (which is noise in our cases)
    Step 2: remove the noise.
    Step 3: Sinse the remaining pixels are either Red (hour hand) or Green (minute
            hand), just check red channel and green channel. If red channel greater
            than other two channels, the pixel is red.The same applies to the green
            channel.
    '''
    
    #### Step 1 ------------------------------------------------------------------------
    
    ## Way 1: Difference method to remove noise, less calculation but need to modify threshold
    ## of noise if applies to other situatiuon. See definetion blow.
    
    noise_positions = noise_position_diff(clock_RGB, threshold_of_noise = 0.15)
    
    ## Way 2: Kmeans method to remove noise, parameter-free but lack of robustness. The
    ## convergence and potential overfitting of K-means are uncertain.

    # noise_positions = noise_position_Kmeans_2(clock_RGB)
    
    #### Step 2 ------------------------------------------------------------------------
    
    ## Separate the three channels.
    Red_layer = clock_RGB[:, :, 0]
    Green_layer = clock_RGB[:, :, 1]
    Blue_layer = clock_RGB[:, :, 2]
    
    ## Remove noises from three channels.
    Red_layer[noise_positions] = 0
    Green_layer[noise_positions] = 0
    Blue_layer[noise_positions] = 0
    
    #### Step 3 ------------------------------------------------------------------------
    
    ## Identify red pixels from remaining pixels.
    Red_position = (Red_layer > Green_layer) & (Red_layer > Blue_layer)
    Hour_hand = np.argwhere(Red_position)
    
    ## Identify green pixels from remaining pixels.
    Green_position = (Green_layer > Red_layer) & (Green_layer > Blue_layer)
    Minute_hand = np.argwhere(Green_position)

    return Hour_hand, Minute_hand
    # return np.stack((Red_layer, Green_layer, Blue_layer), axis = -1)      # for check output conviniently

def noise_position_diff(clock_RGB, threshold_of_noise = 0.18):
    '''
    Return the index of of noise.
    (Way 1, Difference method to remove noise)
    
    Step 1: Separate the three channels. Calculate the absolute difference between each 
            pair of channels for every pixel.
    Step 2: if all the absolute difference of three pairs less than [threshold_of_noise],
            we consider this pixel to have no color (appearing gray).
    '''
    
    #### Step 1 ------------------------------------------------------------------------
    
    ## Separate the three channels.
    Red_layer = clock_RGB[:, :, 0]
    Green_layer = clock_RGB[:, :, 1]
    Blue_layer = clock_RGB[:, :, 2]
    
    ## Calculate the absolute difference between each pair of channels for every pixel.
    diff_red_green = np.abs(Red_layer - Green_layer) < threshold_of_noise
    diff_red_blue = np.abs(Red_layer - Blue_layer) < threshold_of_noise
    diff_blue_green = np.abs(Blue_layer - Green_layer) < threshold_of_noise
    
    #### Step 2 ------------------------------------------------------------------------
    
    ## Identify the pixel dont have color (appearing gray).
    noise_positions = diff_red_green & diff_red_blue & diff_blue_green
    
    return noise_positions

def noise_position_Kmeans_2(clock_RGB):
    '''
    Return the index of of noise.
    (Way 2, Kmeans method to remove noise)
    
    Step 1: Calculate the standard deviation of three channel for every pixel (times data by 
            10 to magnify the std. which will decrease the difficult to identify the noise).
    Step 2: Flatten the standard deviation matrix into a one-dimensional array and use Kmeans
            to devide them into two categories.
    Step 3: Find the category who have more elements, which is the location of noise.
    '''
    
    #### Step 1 ------------------------------------------------------------------------
    
    shape_of_clock = clock_RGB.shape                      # store the origin shape of clock_RGB
    
    clock_RGB = np.array(clock_RGB)                       # make sure the date in type of np.array  
      
    ## Calculate the standard deviation of three channel for every pixel (times by 10 to magnify)
    variance_matrix = np.var(10 * clock_RGB, axis=2)
    
    #### Step 2 ------------------------------------------------------------------------
    
    ## Flatten the standard deviation matrix into a one-dimensional array
    variance_matrix_flatten = variance_matrix.flatten()
    ## Transposition for the requirement of Kmeans input rules
    variance_matrix_flatten = variance_matrix_flatten.reshape(-1, 1)
    ## Build a Kmeans model who have 2 categories
    kmeans_model = KMeans(n_clusters=2)
    ## Applies Kmeans to our model
    kmeans_model.fit(variance_matrix_flatten)
    labels = kmeans_model.predict(variance_matrix_flatten)
    ## Reshape the label to its origin shape
    labels = labels.reshape(shape_of_clock[0], shape_of_clock[1])
    ## Calculate how many "category-one" in the lable
    num_ones = np.sum(labels)
    ## Calculate how many "category-zero" in the lable
    num_zeros = labels.size - num_ones
    
    noise_positions = np.zeros((shape_of_clock[0], shape_of_clock[1]), dtype=bool)
    
    ## Create a logic matrix to store the index of noise.
    if num_ones > num_zeros:               # if "category-one" more than "category-zero"
        noise_row, noise_col = np.where(labels == 1)
    else:                                  # otherwise
        noise_row, noise_col = np.where(labels == 0)
    
    ## Mark the index of noise 
    noise_positions[noise_row, noise_col] = True

    return noise_positions

def get_angle(coords):
    '''
    Return the angle (in radian) for the input hand (coods)

    Step 1: Transformation of coordinates with a rough new origin. [50, 50] with 
            a 10% deviation.
    Step 2: Calculate the distance between each of the two endpoints and the current
            origin [0,0], and set the closer endpoint as the precise new origin [0,0].
    Step 3: Check if the line will vertical to x-axis (which will have no slope)
    Step 4: If the line have slope, calculate it. And determine the angle value based
            on the quadrant in which the hand is located.
    '''

    #### Step 1 ------------------------------------------------------------------------
    
    ## Find the rough new orign.
    ## Figsize is 101 x 101, the rough orgin will have a 10% deviation.
    new_origin = np.zeros((2))    
    new_origin[0] = ((101 - 1) / 2) * (1 - random.uniform(0, 0.01))
    new_origin[1] = ((101 - 1) / 2) * (1 - random.uniform(0, 0.01))
    
    ## The index get form image will lead to a graphic of Fourth quadrant.
    ## We need to Rotate 90 degrees counterclockwise and move the new origin
    ## to the rough orgin.
    x = coords[:, 1] - new_origin[0]
    y = new_origin[1] - coords[:, 0]

    #### Step 2 ------------------------------------------------------------------------
    
    ## Find the index of two endpoint (sort -> first one & last one)
    sort_indices = np.argsort(x)
    endpoint_1 = np.array([x[sort_indices[0]], y[sort_indices[0]]])
    endpoint_2 = np.array([x[sort_indices[-1]], y[sort_indices[-1]]])
    
    ## Norm is Euclidean distance
    distance_1 = np.linalg.norm(endpoint_1)
    distance_2 = np.linalg.norm(endpoint_2)
            
    if distance_1 < distance_2:          # Change origin to endpoint 1
        x -= endpoint_1[0]
        y -= endpoint_1[1]
    elif distance_1 > distance_2:        # Change origin to endpoint 2
        x -= endpoint_2[0]
        y -= endpoint_2[1]     
    else:                                # If two distance are same
        raise ValueError("There is a problem with center point accuracy.") 
    
    #### Step 3 ------------------------------------------------------------------------
    
    ## If the line vertical to x-axis, all its x_index will be the same
    if np.max(x) - np.min(x) == 0:

        ## Check the sign of the element with the largest absolute value in y.
        y_max = y[np.abs(y).argmax()]
        
        if y_max > 0:                    # point to 12 o'clock
            angle = 0
        elif y_max < 0:                  # point to 6 o'clock
            angle = np.pi
        else:
            raise ValueError("The set of y is empty.")
        
    #### Step 4 ------------------------------------------------------------------------
    
    else:
        slope, *_ = linregress(x, y)      
        theta = np.arctan(slope)         # angle with (+)x-axis
        
        x_max = x[np.abs(x).argmax()]
        y_max = y[np.abs(y).argmax()]    # get the vector of the hand
        
        if x_max > 0 and y_max >= 0:     # First quadrant
            angle = np.pi / 2 - np.abs(theta)
        elif x_max > 0 and y_max < 0:    # Fourth quadrant
            angle = np.pi / 2 + np.abs(theta)
        elif x_max < 0 and y_max < 0:    # Third quadrant
            angle = np.pi * 3 / 2 - np.abs(theta)
        elif x_max < 0 and y_max >= 0:   # Second quadrant
            angle = np.pi * 3 / 2 + np.abs(theta)
        
    return angle
    
def analog_to_digital(angle_hour, angle_minute):
    '''
    Returns the time in digital format, as a string.
    '''
    
    ## Convert radians into a specific time
    hour_0 = 12 * angle_hour / (2 * np.pi)
    minute_0 = 60 * angle_minute / (2 * np.pi)
    
    ## A threshold to eliminate computational errors in Python.
    precision = 1e-9
    
    if angle_hour == 0:                               # point to 12 o'clock exactly
        hour = int(12)
    elif angle_hour == np.pi:                         # point to 6 o'clock exactly
        hour = int(6)
    else:
        if np.ceil(hour_0) - hour_0 < precision:      # A rounding error occurred.
            hour = int(np.ceil(hour_0))
        else:
            ## General case, Find the largest integer less than hour_0.
            hour = int(np.floor(hour_0))
    
    ## Transfer 00:XX to 12:XX
    if hour == 0:
        hour = 12
        
    if angle_minute == 0:                             # point to 0 minute exactly
        minute = int(0)
    elif angle_minute == np.pi:                       # point to 30 minute exactly
        minute = int(30)
    else:
        if np.ceil(minute_0) - minute_0 < precision:  # A rounding error occurred.
            minute = int(np.ceil(minute_0))
        else:
            ## General case, Find the largest integer less than minute_0.
            minute = int(np.floor(minute_0))
    
    ## An empty string to store the result    
    time_str = ''
    
    ## If the minute or hour is single-digit number, add a zero before it.
    if hour < 10:
        time_str += '0' + str(hour) + ':'
    else:
        time_str += str(hour) + ':'
        
    if minute < 10:
        time_str += '0' + str(minute)
    else:
        time_str += str(minute)

    return time_str

def check_alignment(angle_hour, angle_minute):
    '''
    Returns the misalignment in minutes.
    
    Core Idea: Calculte the right location of minute hand through hour hand
               and comapre the location of minute shown in image to it.
    
    Output will 
    - be positive when the minute hand is ahead of where it should be
    - be negative when the minute hand is behind of where it should be
    '''
    
    ## Convert radians into a specific time
    hour_0 = 12 * angle_hour / (2 * np.pi)
    minute_0 = 60 * angle_minute / (2 * np.pi)
    
    ## A threshold to eliminate computational errors in Python.
    precision = 1e-9
    
    ## Find the accurate minute
    if angle_hour == 0:                               # point to 12 o'clock exactly
        minute_exact = 0                              # minute hand should point to zero
    elif angle_hour == np.pi:                         # point to 6 o'clock exactly
        minute_exact = 0                              # minute hand should point to zero
    else:
        if np.ceil(hour_0) - hour_0 < precision:      # A rounding error occurred.
            minute_exact = 0
        else:
            ## General case, find the largest integer less than hour_0 and minus it.
            ## The remaing should be time elapsed this hour. Time it with 60 is the
            ## exact minute.
            minute_exact = int(60 * (hour_0 - np.floor(hour_0)))
    
    ## Find the shown minute
    if angle_minute == 0:                             # point to 0 minute exactly
        minute_leg = 0
    elif angle_minute == np.pi:                       # point to 30 minute exactly
        minute_leg = 30
    else:
        if np.ceil(minute_0) - minute_0 < precision:  # A rounding error occurred.
            minute_leg = int(np.ceil(minute_0))
        else:
            ## General case, find the largest integer less than minute_0
            minute_leg = int(minute_0)
    
    ## Subtract two minutes
    diff_minute = minute_leg - minute_exact
    
    ## Misalignment will never be more than 30 minutes. If happen, it indecate a 
    ## pointer has completed an extra rotation.
    if diff_minute >= 30:
        diff_minute = diff_minute - 60
    elif diff_minute <= -30:
        diff_minute = diff_minute + 60
    
    return diff_minute

def validate_batch(folder_path, tolerance):
    '''
    This function will write a '.txt' file called batch_X_QC.txt (where X will be replaced
    by the batch number) for all the images in the given folder, containing the following
    information:
    
    Batch number: [X]
    Checked on [date and time]

    Total number of clocks: [X]
    Number of clocks passing quality control ([X]-minute tolerance): [X]
    Batch quality: [X]%

    Clocks to send back for readjustment:
    clock_[X]   [X]min
    clock_[X]   [X]min
    clock_[X]   [X]min
    [etc.]
    '''
    
    ## Creat two empty list to store the difference and number of a clock
    diff_list = []
    num_list = []
    
    ## Loop in the files of input folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):                # Make sure the file is a PNG file
            
            file_path = os.path.join(folder_path, file_name)
            clock_RGB = read_image(file_path, show=False)            # Read image
            hour_pixels, minute_pixels = get_clock_hands(clock_RGB)  # Get clock hand
            angle_hour = get_angle(hour_pixels)                      # Get angle of hour
            angle_minute = get_angle(minute_pixels)                  # Get angle of minute
            diff = check_alignment(angle_hour, angle_minute)         # Calculate the differnce

            diff_list.append(diff)                                   # Add the diff into its list
            num_list.append(re.search(r'\d+', file_name).group())    # Add corresponding number into its list
            
        diff_array = np.array(diff_list)
        num_array = np.array(num_list)                               # Transfer list into np array
        pass_count = len(np.where(abs(diff_array) <= tolerance)[0])  # Count the not pass Clock
        
        not_pass_indices = np.where(abs(diff_array) > tolerance)[0]  # Mark the not pass Clock
        not_pass_array = diff_array[not_pass_indices]
        not_pass_num = num_array[not_pass_indices]                   # Store them
        
        abs_not_pass = abs(not_pass_array)
        sorted_not_pass_indeices = np.argsort(-abs_not_pass)
        
        not_pass_indices = not_pass_indices[sorted_not_pass_indeices]
        not_pass_array = not_pass_array[sorted_not_pass_indeices]
        not_pass_num = not_pass_num[sorted_not_pass_indeices]        # Sort in descending order.

        output_path = 'QC_reports'      # Create an output folder (if it doesn't exist, create one).
        if not os.path.exists(output_path):
            os.makedirs(output_path)            
        
        ## Name the TXT file.
        txt_name = os.path.join(output_path, f'batch_{folder_path[-1]}_QC.txt')
        
        ## Write the information to the TXT file.
        with open(txt_name, "w") as file:
            file.write(f'Batch number: {folder_path[-1]}\n')
            
            now = datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d, %H:%M:%S")
            file.write(f'Checked on {formatted_datetime}\n\n')
            
            file.write(f'Total number of clocks: {len(diff_list)}\n')
            file.write(f'Number of clocks passing quality control ({tolerance}-minute tolerance): {pass_count}\n')
            file.write(f'Batch quality: {(pass_count / len(diff_list)):.1%}\n\n')
            file.write(f'Clocks to send back for readjustment:\n')
            
            for i in range(len(not_pass_array)):
                                
                num_of_clock = not_pass_num[i]
                diff_of_clock = not_pass_array[i]

                file.write(f'clock_{num_of_clock:<2}   {diff_of_clock:+3} min\n')

def get_angle_from_path(path):
    '''
    Return the angle of hour hand and minute hand
    '''
    
    ## Just like what was done in the previous function.
    clock_RGB = read_image(path, show=False)
    hour_pixels, minute_pixels = get_clock_hands(clock_RGB)
    angle_hour = get_angle(hour_pixels)
    angle_minute = get_angle(minute_pixels)
    return angle_hour, angle_minute

def check_coupling(path_1, path_2):
    '''
    return a string dexcribe the coupling situation of a clock
    
    Step 1: Use two input hour hand to calculate exact how long has passed.
    Step 2: Calculate where the minute hand should located now if the clock dont have a
            coupling problem.
    Step 3: Compare where the minute shown to result of step 3, and calculate the coupling
            error. 
    '''
    
    #### Step 1 ------------------------------------------------------------------------
    
    angle_hour_1, angle_minute_1 = get_angle_from_path(path_1)
    angle_hour_2, angle_minute_2 = get_angle_from_path(path_2)
    
    ## Calculate the passed angle of hour hand
    exact_hour_past_angle = angle_hour_2 - angle_hour_1
    
    ## If hour_hand 2 < hour_hand 1, which indecate it has complete a full round
    if exact_hour_past_angle < 0:
        exact_hour_past_angle += 2 * np.pi
    
    ## Calculate exact past time through passed angle of hour hand.
    exact_past_time = 12 * (exact_hour_past_angle / (2 * np.pi))
    
    #### Step 2 ------------------------------------------------------------------------
    
    ## The hour hand completes one full rotation, while the minute hand completes twelve
    ## rotations.
    right_current_minute_angle = (angle_minute_1 + exact_hour_past_angle * 12) % (2 * np.pi)
    
    minute_exact = 60 * right_current_minute_angle / (2 * np.pi)
    minute_show = 60 * angle_minute_2 / (2 * np.pi)
    
    #### Step 3 ------------------------------------------------------------------------
    
    ## Calculate difference
    diff = minute_show - minute_exact
    
    ## Difference will never be more than 30 minutes. If happen, it indecate a pointer
    ## has completed an extra rotation.
    if diff >= 30:
        diff -= 60
    elif diff <= -30:
        diff += 60
    
    ## Express the difference in terms of hours.
    diff = diff / exact_past_time
    
    diff_minute = int(abs(diff))
    diff_second = int(60 * (abs(diff) - diff_minute))
    
    precision = 1e-5     # Give a error of tolerence
    
    ## Write down the report.
    if abs(diff) < precision:
        result = 'The hour and minute hand are coupled properly.'
    elif diff > 0:       
        result = f'The minute hand gains {diff_minute:>2} minutes, {diff_second:>2} seconds per hour.'
    else:
        result = f'The minute hand loses {diff_minute:>2} minutes, {diff_second:>2} seconds per hour.'

    return result





## For test.
if __name__ == "__main__":
    
    import quality_control as qc   
       
    pass