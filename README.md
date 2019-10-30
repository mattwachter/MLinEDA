# Prediction of detailed routing errors based on a placed layout via machine learning.
The aim of this project is to predict shorts that occur after detailed routing
as part of an electronic design automation (EDA) flow.

Several software tools needed to be scripted to accomplish that goal.
There are scripts that control the layout viewer and editor [Klayout](https://www.klayout.de/), the placement tool [Eh?Placer](https://www.ucalgary.ca/karimpour/node/10), the (commercial) place & route suite _Cadence ® Encounter ® Digital Implementation System_ and the [Keras](https://keras.io/) neural network framework.

Additional information to the one below can be found in the comments at the start of the individual scripts.


## Klayout
`Klayout/klayout_scripts.py` provides functionality to import a  GDSII (`.gds`) or OASIS (`.oas`) layout file, separately save its layers, and save square sectors of it as image (`.png`) files.

## Eh?Placer and Encounter
The Benchmarks from the [ISPD 2015 Blockage-Aware Detailed Routing-Driven Placement Contest](http://www.ispd.cc/contests/14/ispd2015_contest.html) were placed with the [Eh?Placer](https://www.ucalgary.ca/karimpour/node/10) with the following command executed in the folder of each benchmark:
```sh
~/bin/EhPlacer_ispd01.placer -tech_lef ./tech.lef -cell_lef ./cells.lef -floorplan_def ./floorplan.def -output ./EhPlacerOutput.def -placement_constraints ./placement.constraints -verilog ./design.v -cpu 8
```
The commands in the shell script `Encounter/place_route_get_info.sh` execute the command above and the TCL scripts that control the Encounter software to import and save the LEF/DEF file output of Eh?Placer as an Encounter file (`import_save.tcl`), route the design ('route.tcl') and then extract information from the placed (to use as features) and routed (to use as labels for the neural network) design to save the info as CSV files (`get_info.tcl`).

## Keras
### Feature Extraction
`extract_data.py` extracts and processes the information saved as `.csv` files by `get_info.tcl` and should therefore be used in the same folder.
The generated data should be saved in the `Data` folder.
The output is a pickled list of numpy arrays with the training data (e.g. `design_placed_training_data_mgc_fft_1.pickle`) and the labels (e.g. `design_routed_labels_shorts_mgc_fft_1.pickle`).
### Configuring and training the neural network
The data extracted from the layouts via the TCL scripts is imported and
processed by `neural_network.py`.

#### Evaluation
The history data how the loss and accuracy metrics have changed during training is plotted with `metrics_history.py`.
`metrics.py` calculates different custom metrics based on the predictions of the fitted (trained) neural network
They help with the evaluation of binary predictors for unbalanced (e.g. way more 0s than 1s) data.
Its output is also saved as CSV files.

#### Google Colab
There are variants of `neural_networks.py` for use as Jupyter Notebooks in [Google Colab](design_placed_training_data_mgc_fft_1.pickle) with support for either GPU or TPU (Tensor Processing Unit) acceleration.
The GPU acceleration works flawlessly already, while the TPU version will probably work once Tensorflow 2.1 is released.

#### Custom metrics
There is currently the problem that the `functions metrics_K()` and
`binary_confusion_matrix_K()` do not calculate correct results.
Nevertheless, they are still included in `neural_network-experimental_keras_binary_predictor_quality_metrics.py` to give the opportunity
to fix these metrics. They only use the keras.backend functions and can since
be called during model.fit() training process.

####  Draft to use Talos for automated optimization of hyper parameters
With more processing power, `hyperparameter_optimization_talos.py` can be used to experiment with parameters such as different batch sizes, optimization algorithms, layer shapes and training epochs.

## Data
Empty folders that correspond to the data structures in the above scripts are
set up in the `Data` folder.
Due to licensing issues with the
Encounter Digital Implementation Suite no output derived from it can be included
in this public repository.

## Style guide

The [Google Python style guide ](http://google.github.io/styleguide/pyguide.html)
    was used.

## Project origins
This project was created as part of a _Studienarbeit_  (German equivalent of a Bachelor's thesis) by Matthias von Wachter with supervision by
[Dr. Robert Fischbach](https://www.ifte.de/mitarbeiter/fischbach.html) at the
[Institute of Electromechanical and Electronic Design](https://www.ifte.de/english/index.html) at the [Dresden University of Technology](https://tu-dresden.de/).

## License
This software is licensed under the [MIT License](https://mit-license.org/).
The PDF with the (German) paper `rpml.pdf` "Entwicklung eines maschinellen Lernverfahrens basierend auf neuronalen Netzen zur Vorhersage der Verdrahtbarkeit platzierter Schaltungslayouts" is licensed under the [Creative Commons Attribution-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/deed.de).