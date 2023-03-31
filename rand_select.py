import random

def randomly_select(arr, x_percent=25):

    # Calculate the total number of coordinates in the array
    total_coords = len(arr) * len(arr[0])

    # Calculate the number of coordinates to select based on the percentage
    num_coords_to_select = int((x_percent/100) * total_coords)

    # Create an empty list to store the selected coordinates
    selected_coords = []

    # Loop until we have selected the desired number of coordinates
    while len(selected_coords) < num_coords_to_select:
        # Choose a random row and column index
        rand_row_idx = random.randint(0, len(arr)-1)
        rand_col_idx = random.randint(0, len(arr[0])-1)
        
        # Check if this coordinate has already been selected
        if (rand_row_idx, rand_col_idx) not in selected_coords:
            # If not, add it to the selected coordinates list
            selected_coords.append((rand_row_idx, rand_col_idx))

    # Now we can loop through the selected coordinates and do whatever we want with them
    for coord in selected_coords:
        row_idx, col_idx = coord
        arr[row_idx][col_idx] = 1
    
    return arr