""" These scripts for Klayout (https://klayout.de)
save the layout's separate layers,
calculate the positions of the boxes we want to cut out depending on the
dimensions of the layout and the desired length of the size of the sqaures,
and then save all of those squares (optionally) as layout files and as
PNG image files.
TODO Usage instructions in README.md
After installing Klayout on your system
(https://www.klayout.de/staging/build.html)
Run the following shell command with the path to this script:
klayout -r klayout_scripts.py
"""
# db is the module for the python binding of Klayout
# import klayout.db as db
import pya
# Create a folder structure for the files we create
import os
# Save the Python list of PNG file names on disk
import pickle


# ************************** Config ***************************************
fname = "./Data/TestKlayoutScript/design_placed.oas"
box_size = 18000  # int (Assume that dbu = 0.001 micro meter)
box_size_pixel = 540
box_size_pixel_layer = 1000
save_all_layers = False

# Save all layers corresponding to a certain mask
layers_merged = \
    [
        ["metal1", [18, 6, 9]],
        ["metal2", [4, 10, 19]],
        ["metal3", [7, 3, 12]],
        ["metal4", [2, 5, 14]],
        ["metal5", [8, 16]],
        ["COMP", [1]]
    ]
# ************************** Config ***************************************


def split_file_name(fname):
    """Splits a file name into path, title and ending and returns them.

    Args:
        fname (str): File with full or relative path
    Returns:
        string: fname_path
        string: fname_title
        string: fname_ending
    """
    # Make sure we have an absolute path, not a relative path.
    fname = os.path.abspath(fname)
    # Split into head and tail(part at the end without slashes)
    (fname_path, fname_head) = os.path.split(fname)
    fname_parts = fname_head.split(".")
    fname_title = fname_parts[0]
    fname_ending = fname_parts[1]  # E.g. gds
    print("Folder path:", fname_path)
    print("File title:", fname_title)
    print("File ending:", fname_ending)
    return (fname_path, fname_title, fname_ending)


# TODO clear folder of existing .gds, .oas, and .png files
def create_folder_structure(fname_path, fname_title):
    """Creates a new folder and returns its path.

    Args:
        fname_path: Path of containing folder (string)
        fname_title: Name of new folder (string)

    Returns:
        new_folder_path: Path of created folder (string)
    """
    # make sure we have an absolute path, not a relative path
    fname_path = os.path.abspath(fname_path)
    # New folder path with trailing slash
    new_folder_path = os.path.join(fname_path, fname_title, "")
    try:
        os.makedirs(new_folder_path)
        print("Created new folder '" + new_folder_path + "'.")
    except FileExistsError:
        print("Folder '" + new_folder_path +
              "' exists already, skipping folder creation.")
    return new_folder_path


# TODO Raise errors for invalid layout files, string type, int size...
def save_as_png(fname_layout, fname_png, box_size_pixel):
    """Loads a layout file and saves it as a PNG image.

    Args:
        fname: Name of input OASIS or GDSII layout file (string)
        fname_png: Name of output PNG file (string)
        box_size_pixel: Length of the sides of the PNG image in pixels (int)
    """
    # Create new LayoutView to use save_image_with_options()
    # Fixed thanks to
    # https://www.klayout.de/forum/discussion/1266/segfault-when-calling-layoutview-in-python-script  # noqa: E501
    main_window = pya.MainWindow.instance()
    box_empty = pya.Box()   # Needed for save_layout_with_options()
    # Load the layout box we just saved.
    # Replace the current layouts (mode 0), into a new view
    # (mode 1) or add the layout to the current view (mode 2).
    # In mode 1, the new view is made the current one.
    cell_view_target = main_window.load_layout(fname_layout, 0)
    # type: pya.LayoutView()
    layout_view_target = cell_view_target.view()
    # method of LayoutView:
    # void save_image_with_options (string filename,
    #    unsigned int width,unsigned int height,
    #    int linewidth,int oversampling,
    #    double resolution,const DBox target,
    #    bool monochrome)
    layout_view_target.save_image_with_options(
        fname_png, box_size_pixel, box_size_pixel,
        0, 0, 0, box_empty, 0)
    print("Saved file", fname_png)


def save_separate_layers(fname, box_size_pixel, layers_merged=None,
                         save_all_layers=True):
    """Loads a layout file and saves its layers as separate files.

    The layout file format can be GDSII or Oasis.
    A PNG image sides of box_size_pixel will also be saved.
    The resulting folderstructure of 'mapv3.gds' with 2 layers will be:
    |-- map9v3
    |  |-- map9v3_layer_00
    |-- map9v3_layer_00.gds
    |  |-- map9v3_layer_01
    |-- map9v3_layer_01.gds
    Optionally, the mask layers consisting of multiple layers  are also
    saved if 'layers_merged' is passed.

    Args:
        fname: File name of layout (string)
        box_size_pixel: Length of the sides of the PNG image in pixels (int)
        layers_merged: List of layers per layerObjName
            e.g. [["mask1", [0, 1]],
                  ["mask2", [2, 3]]] ([[string[int]]])
        save_all_layers: Whether all layers should also be saved as
            single files (Bool)

    Returns:
        fnames_layer: List of saved layers (strings)
    """
    # Create a folder structure for the many files we will be creating
    (fname_path, fname_title, fname_ending) = split_file_name(fname)
    new_folder_path = create_folder_structure(fname_path, fname_title)
    fname_path = new_folder_path

    # Klayout class Layout contains cell hierarchy
    # and manages cell and layout names
    layout = pya.Layout()
    # Controls reading of layout files
    load_opt = pya.LoadLayoutOptions()

    dbu = layout.dbu        # Data base unit
    print("The data base unit(dbu) of layout",
          fname_title + "." + fname_ending, "is", str(dbu), "micro meters.")

    # Settings for loading of layout file
    load_opt.select_all_layers()

    # Map logical layer indices to phyical layer names.
    layer_map = layout.read(fname, load_opt)

    # Get list of logical layers detected in the file.
    layer_indices = layout.layer_indices()

    fnames_layer = []

    if save_all_layers:
        layer_mappings = []
        for i in layer_indices:
            # Two digits with leading zero
            layer_index_str = "{:02d}".format(layer_indices[i])
            fname_layer = fname_path + fname_title + \
                "_layer_" + layer_index_str + "." + fname_ending
            fnames_layer.append(fname_layer)
            # Get LayerInfo object for current layer with physical layer
            properties = layout.get_info(i)
            save_opt = pya.SaveLayoutOptions()
            save_opt.deselect_all_layers()
            # Save only one layer (i)
            save_opt.add_layer(layer_indices[i], properties)
            layout.write(fname_layer, save_opt)
            # Save the physical layer mapped to logical layer i as a string
            layer_map_str_i = layer_map.mapping_str(layer_indices[i])
            # The first part contains the information about
            # the physical layer (layer number/type)
            layer_phy = layer_map_str_i.split(" ")[0]
            layer_mappings.append(str(layer_indices[i]) + "," + layer_phy)
            print("Saving the physical layer", layer_phy,
                  "mapped to logical layer", str(layer_indices[i]), "as",
                  fname_layer)
            # Load the saved layout files from disk and save as PNG:
            fname_layer_png = fname_path + fname_title + \
                "_layer_" + layer_index_str + ".png"
            save_as_png(fname_layer, fname_layer_png, box_size_pixel)
        # Save mapping to text file
        # Save layer map in same directory as original layout file.
        fname_layer_map = fname_path + fname_title + "_layermap.csv"
        with open(fname_layer_map, 'w') as f:
            f.write("Logical layer, Physical layer\n")
            for m in layer_mappings:
                f.write(m + "\n")
        print("The mappings from logical to physical layer were saved in",
              fname_layer_map)

    if layers_merged is not None:
        # Save all layers corresponding to a certain mask
        # as one merged layout file
        # TODO Solve without unnecessary copy/paste
        for lm in layers_merged:
            layerObjName = lm[0]
            fname_layer_merged = fname_path + fname_title + \
                "_layer_" + layerObjName + "." + fname_ending
            fnames_layer.append(fname_layer_merged)
            # Get LayerInfo object for current layer with physical layer
            save_opt = pya.SaveLayoutOptions()
            save_opt.deselect_all_layers()
            for i in lm[1]:
                properties = layout.get_info(i)
                save_opt.add_layer(layer_indices[i], properties)
            layout.write(fname_layer_merged, save_opt)
            # Load the saved layout files from disk and save as PNG:
            fname_layer_merged_png = fname_path + fname_title + \
                "_layer_" + layerObjName + ".png"
            save_as_png(fname_layer_merged, fname_layer_merged_png,
                        box_size_pixel)

    return fnames_layer    # String[]


# ***************partition_layout*************************


def biggest_bounding_box(fname):
    """Finds the bounding box enclosing all cells in all layers of a layout.

    This dimension of the whole layout and its data base unit is returned.


    Args:
        fname: Layout file name (string)

    Returns:
        biggest_bounding_box: Box enclosing the whole layout (pya.Box())
        dbu: Data base unit of the layout (double)
    """
    layout = pya.Layout()
    layout.read(fname)
    biggest_bounding_box = pya.Box()
    for top_cell_index in layout.each_top_cell():
        top_cell = layout.cell(top_cell_index)  # Cell object from index
        bbox = top_cell.bbox()  # Bounding box of pya.Cell object
        # + operator joins Boxes into Box enclosing both inputs.
        biggest_bounding_box = biggest_bounding_box + bbox
    dbu = layout.dbu  # Data base unit
    return (biggest_bounding_box, dbu)


# TODO Put functionality for calculating boxes[] into separate function.
# This would enable doing the calculations just once per layout.
def partition_layout(fname, biggest_bounding_box, box_size, box_size_pixel,
                     dbu):
    """Partition layout into squares of equal size and save them.

    Args:
        fname: file name of a single layer Layout (string)
        layout_dimensions: Size of layout (pya.Box)
        box_size: Length of the squares in dbu (data base units) (int)
        box_size_pixel: Length of the sides of a PNG image in pixels (int)
        biggest_bounding_box: Size of the whole layout (pya.Box())
        dbu: Data base unit of whole layout (double)

    Returns:
        png_fnames: List of the file names of the PNG images (strings)
    """
    # Create new valid file names derived from fname.
    (fname_path, fname_title, fname_ending) = split_file_name(fname)
    # Create a folder structure to file away the many files we will be creating
    new_folder_path = create_folder_structure(fname_path, fname_title)
    fname_path = new_folder_path
    layout = pya.Layout()
    layout.read(fname)
    # Copy the LayerInfo properties of the first (and, if layer
    # splitting worked, only) layer of the original layout
    layerinfo_target = layout.get_info(0).dup()
    # Adapt dbu of target layer
    layerinfo_target.dbu = dbu
    # Set Oasis compression level.
    save_opt = pya.SaveLayoutOptions()
    # TODO: Effect of compression level?!
    save_opt.oasis_compression_level = 2    # 0, 1, or 2; 2 is highest
    save_opt.oasis_recompress = True
    # Use size of the whole layout to calculate box positions
    layout_dim_width = biggest_bounding_box.width()  # Returns unsigned int
    layout_dim_height = biggest_bounding_box.height()
    # Number of squares on x axis, + 1 to include overlap
    n_i = layout_dim_width // box_size + 1  # Floor division
    print(n_i, "boxes on x axis")
    # Number of squares on y axis, + 1 to include overlap
    n_j = layout_dim_height // box_size + 1  # Floor division
    print(n_j, "boxes on y axis")
    # Initialize 2D list with 'list comprehension' pattern
    boxes = [[j for j in range(n_j)] for i in range(n_i)]
    # TODO Check indices of list comprehension
    # The origin of the coordinate system is on the bottom left.
    x_start = biggest_bounding_box.left  # int
    y_start = biggest_bounding_box.bottom  # int
    x = x_start
    # i runs from 0 to the number of squares in i direction minus 1
    for i in range(n_i):
        y = y_start
        for j in range(n_j):
            left = x
            bottom = y
            right = x + box_size
            top = y + box_size
            # Box[] boxes with sides of size box_size
            # for saving layout files
            boxes[i][j] = pya.Box(left, bottom, right, top)
            y += box_size
        x += box_size
    # Save all file names of the png images of one layer in a list.
    png_fnames = []
    for top_cell_index in layout.each_top_cell():
        # Flatten all top cells (probably just one) to save
        # CPU time for the multi_clip_into() operation
        # cell_index i, all levels, prune (remove orphan cells)
        layout.flatten(top_cell_index, -1, True)
        # Iterate over x axis of 2D array Box[][] boxes
        for i in range(n_i):
            # Iterate over y axis
            for j in range(n_j):
                # Create new layout to copy clips into
                layout_target = pya.Layout()
                # Insert layer with same properties as the original layer
                layout_target.insert_layer(layerinfo_target)
                layout.clip_into(top_cell_index, layout_target, boxes[i][j])
                fname_box = fname_path + fname_title + "_box_i" + str(
                    i) + "_j" + str(j) + "." + fname_ending
                layout_target.write(fname_box, save_opt)
                print("Saved file", fname_box)
                # Load the saved files from disk and save as PNG:
                fname_box_png = fname_path + fname_title + "_box_i" + str(
                    i) + "_j" + str(j) + ".png"
                save_as_png(fname_box, fname_box_png, box_size_pixel)

                png_fnames.append(fname_box_png)
    return png_fnames


def main():
    """Separate input file into layers and save cut outs from each of them.

    Uses the functions created above to save the layout's separate layers,
    calculate the positions of the boxes we want to cut out depending on the
    dimensions of the layout and the desired length of the size of the squares,
    and then saves all of those squares (optionally) as layout files and as
    PNG image files.
    """
    fnames_layer = save_separate_layers(fname, box_size_pixel_layer,
                                        layers_merged, save_all_layers)
    # Returns pya.Box(), double
    (layout_dimensions, dbu) = biggest_bounding_box(fname)
    print("The biggest bounding box of layout", fname, "is:",
          str(layout_dimensions), "*", dbu, "micro meters.")

    # Save all file names of the PNG images of all layers.
    fnames_png_layers = []
    for fn in fnames_layer:
        fnames_png_layer = partition_layout(fn, layout_dimensions,
                                            box_size, box_size_pixel, dbu)
        # 2D list with dimension len(fnames_layer)*(n_i*n_j)
        fnames_png_layers.append(fnames_png_layer)

    # Save the the Python list of all PNG file names to disk.
    (fname_path, fname_title, fname_ending) = split_file_name(fname)
    # Pattern from Examples in
    # https://docs.python.org/3.7/library/pickle.html
    fname_png_pickle = os.path.join(fname_path, fname_title
                                    + '_png_.pickle')
    with open(fname_png_pickle, 'wb') as f:
        pickle.dump(fnames_png_layers, f, pickle.DEFAULT_PROTOCOL)

    # Make sure we close the MainWindow/GUI Application
    app = pya.Application.instance()
    app.exit(0)     # Performs cleanup, updates config file.


# Do not execute main() when imported as module
if __name__ == '__main__':
    main()
