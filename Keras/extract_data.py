# Encounter/get_info.tcl and to a list of numpy arrays containing the
# data of a box and its neighbors. Save as a binary pickle file.

import numpy as np
import csv
import pickle
import math


def get_box_neighbors(i_cur, j_cur, i_max, j_max):
    """ Return tupels with coordinates of neighboring boxes.

    Raises:
        ValueError:     If input coordinates are outside of design

    Args:
        i_cur(int or float):  index of box in x direction
        j_cur(int or float):  index of box in y direction
        i_max(int or float):  index of outermost box in x direction
        j_max(int or float):  index of outermost box in y direction

    Returns:
        list containing 8 tupels with coordinates of neighboring boxes,
        in clockwise order, starting with 'upper'=(i, j+1):
        (upper, upper_right, right, lower_right, lower, lower_left,
            left, upper_left)
"""
    # Check if outside of grid
    if ((i_cur > i_max) or (i_cur < 0) or (j_cur > j_max) or (j_cur < 0)):
        raise ValueError

    upper = (i_cur, j_cur + 1)
    if (upper[1] > j_max):
        upper = (-1, -1)

    upper_right = (i_cur + 1, j_cur + 1)
    if ((upper_right[0] > j_max) or (upper_right[1] > j_max)):
        upper_right = (-1, -1)

    right = (i_cur + 1, j_cur)
    if (right[0] > j_max):
        right = (-1, -1)

    lower_right = (i_cur + 1, j_cur - 1)
    if (lower_right[0] > i_max) or (lower_right[1] < 0):
        lower_right = (-1, -1)

    lower = (i_cur, j_cur - 1)
    if (lower[1] > j_max):
        lower = (-1, -1)

    lower_left = (i_cur - 1, j_cur - 1)
    if ((lower_left[0] < 0) or (lower_left[0] < 0)):
        lower_left = (-1, -1)

    left = (i_cur - 1, j_cur)
    if (left[0] < 0):
        left = (-1, -1)

    upper_left = (i_cur - 1, j_cur + 1)
    if ((upper_left[0] < 0) or (upper_left[1] > j_max)):
        upper_left = (-1, -1)

    # Return tupels in clockwise order, starting with 'upper'
    return [upper, upper_right, right, lower_right, lower, lower_left,
            left, upper_left]


def get_box_index(i_cur, j_cur, i_max, j_max):
    """ Return the location of the box (index n) in the design array.
    Args:
        i_cur(int or float):  index of box in x direction
        j_cur(int or float):  index of box in y direction

    Returns:
        n(int):               index of box in design_array
                              '-1' if i_cur and j_cur are '-1'
    """
    if ((i_cur == -1) and (j_cur == -1)):
        n = -1
    else:
        n = i_cur * (j_max + 1) + j_cur - 1  # index starts with zero
    return int(n)


def get_box_centrality(i_cur, j_cur, i_max, j_max):
    """ Returns the geometric distance of the box to the center.
    Args:
        i_cur(int or float):  index of box in x direction
        j_cur(int or float):  index of box in y direction

    Returns:
        n(int):               index of box in design_array
                              '-1' if i_cur and j_cur are '-1'
    """
    d = math.sqrt((math.pow(i_cur - i_max/2, 2))
                  + (math.pow(i_cur - i_max/2, 2)))
    return d


def get_num_shorts(string_list):
    """ Returns the number of occurences of 'Short' in an input string list.
    Args:
        string_list(list of string objects)
    Returns:
        numShorts(int):       Number of occurences of 'Short'

    """
    numShorts = 0
    for marker in string_list:
        if (marker == 'Short'):
            numShorts += 1
    return numShorts


def main():
    # Instruct the reader to convert all non-quoted fields to type float.
    csv.QUOTE_NONNUMERIC = True

    design_list = []
    # Read overview CSV file specific to each layout
    with open('reports/placed/box_info.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Append ordered dict to list
            design_list.append(row)

    # Find i and j max
    i_max = 0
    for row in design_list:
        if (int(row['i']) >= i_max):
            i_max = int(row['i'])
        # i and j increase monotonically, and then start again from 0.
        else:
            break
    j_max = 0
    for row in design_list:
        if (int(row['j']) >= j_max):
            j_max = int(row['j'])
        # i and j increase monotonically, and then start again from 0.
        else:
            break
    print('i_max:', i_max)
    print('j_max:', j_max)

    # Extract and preprocess data from individual boxes
    for box in design_list:
        fn = 'reports/placed/dbGet/' + box['box_name'] + '.csv'
        box_data = []
        with open(fn, newline='') as csvfile:
            # This CSV file has each data type in a row, not in a collumn
            reader = csv.DictReader(csvfile, fieldnames=['fieldname'],
                                    restkey='val_list')
            for row in reader:
                # The last element of each row is None.
                # We need to cut it out
                row['val_list'] = row['val_list'][0:-2]  # 0 to second to last
                box_data.append(row)
        # Put the data in more efficient numpy arrays
        instTerms_pt_x = np.asarray(box_data[0]['val_list'], dtype=np.single)
        instTerms_pt_y = np.asarray(box_data[1]['val_list'], dtype=np.single)
        instsArea = np.asarray(box_data[4]['val_list'], dtype=np.single)
        # Calculate the standard deviation of the pin positions
        # Set std_dev zero if there are less than two elements
        if (len(instTerms_pt_x) <= 2):
            instTerms_stddev_x = 0
        else:
            instTerms_stddev_x = np.std(instTerms_pt_x)
        if (len(instTerms_pt_y) <= 0):
            instTerms_stddev_y = 0
        else:
            instTerms_stddev_y = np.std(instTerms_pt_y)

        # Calculate the combined area of all instances in the tile
        instsAreaSum = np.sum(instsArea)

        # Get the geometric distance from the center to each box
        i = int(box['i'])
        j = int(box['j'])
        dist_center = get_box_centrality(i, j, i_max, j_max)

        # Insert the calculated values into the ordered dict of the current box
        box['instTerms_stddev_x'] = instTerms_stddev_x
        box['instTerms_stddev_y'] = instTerms_stddev_y
        box['instsAreaSum'] = instsAreaSum
        box['dist_center'] = dist_center

    # Create numpy array from the list of OrderedDict 'design_list'
    num_boxes = len(design_list)
    array_fields = ['numInsts', 'numInstTerms', 'numLocalNets',
                    'numGlobalNets', 'instTerms_stddev_x',
                    'instTerms_stddev_y', 'instsAreaSum', 'dist_center']
    print('Saving array with the following data fields:\n', array_fields)
    # Unfilled boxes are '-1'; Leave out the 'name' field
    design_array = np.full((num_boxes, len(array_fields)), -1, dtype=np.single)

    # print('design_list:\n', design_list)
    # print('design_array:\n', design_array)
    for a in range(num_boxes):
        for b in range(len(array_fields)):
            design_array[a, b] = design_list[a][array_fields[b]]

    # Get the neighboring boxes for each box
    # Generate a row with all zeros as standin for boxes outside design
    row_outside = np.zeros(len(design_array[0, :]), dtype=np.single)

    # Generate a list (TODO: or tensor?!) of arrays containing the information
    # of a box (first) and of its 8 neighbors.
    boxes_placed_with_neighbors = []
    print(design_array.shape)
    for a in range(len(design_array)):
        i = int(design_list[a]['i'])
        j = int(design_list[a]['j'])
        box_array = np.full((9, len(array_fields)), -1, dtype=np.single)
        box_array[0, :] = design_array[a, :]
        neighbors = get_box_neighbors(i, j, i_max, j_max)
        # print(neighbors)
        neighbor_indices = []
        for neigh in neighbors:
            index = get_box_index(neigh[0], neigh[1], i_max, j_max)
            neighbor_indices.append(index)
        # print(neighbor_indices)
        for b in range(8):  # 8 neighbors
            n = neighbor_indices[b]
            # Set all values to zero if neighbor box is outside design
            if (n == -1):
                box_array[b+1, :] = row_outside  # first row is center box
            else:
                box_array[b+1, :] = design_array[n, :]
        # print('box_array:\n', box_array)
        boxes_placed_with_neighbors.append(box_array)

    # Save the the Python list of all box with neighbor arrays to disk.
    # See https://docs.python.org/3.7/library/pickle.html
    fname_pickle = './reports/design_placed_box_arrays_neighbors.pickle'
    with open(fname_pickle, 'wb') as f:
        pickle.dump(boxes_placed_with_neighbors, f, pickle.DEFAULT_PROTOCOL)

    # Get number of "Short" violations in routed design
    # reuse box_name parameter in (placed) 'design_list'
    box_data_routed = []
    for box in design_list:
        fn = 'reports/routed/dbGet/' + box['box_name'] + '.csv'
        box_data = []
        with open(fn, newline='') as csvfile:
            # This CSV file has each data type in a row, not in a collumn
            reader = csv.DictReader(csvfile, fieldnames=['fieldname'],
                                    restkey='val_list')
            for row in reader:
                box_data = row['val_list']
        box_data_routed.append(box_data)

    # Save the number of shorts in each box in a (small, 1D array) in a list
    boxes_routed_markers = []
    numShorts_total = 0
    # Save wether there are shorts in each box or not.
    boxes_routed_shorts_bool = []
    for box in box_data_routed:
        marker_subtypes = box
        numShorts = get_num_shorts(marker_subtypes)
        numShorts_total += numShorts
        boxes_routed_markers.append(np.asarray(numShorts, dtype=np.single))
        if (numShorts == 0):
            boxes_routed_shorts_bool.append(0)
        else:
            boxes_routed_shorts_bool.append(1)

    print('Total number of Shorts found:', numShorts_total)
    print('length boxes_routed_markers:', len(boxes_routed_markers))

    # Save the the Python list of all box with neighbor arrays to disk.
    # See https://docs.python.org/3.7/library/pickle.html
    fname_pickle = './reports/design_routed_box_arrays_markers.pickle'
    with open(fname_pickle, 'wb') as f:
        pickle.dump(boxes_routed_markers, f, pickle.DEFAULT_PROTOCOL)
    # Save variant with bools instead of absolute numbers
        fname_pickle = './reports/design_routed_box_arrays_shorts_bool.pickle'
    with open(fname_pickle, 'wb') as f:
        pickle.dump(boxes_routed_shorts_bool, f, pickle.DEFAULT_PROTOCOL)


# Do not execute main() when imported as module
if __name__ == '__main__':
    main()
