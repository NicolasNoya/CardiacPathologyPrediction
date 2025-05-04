# CardiacPathologyPrediction
This is the README of the CardiacPathologyPrediction project made by Francisco Nicolás Noya in the context of the 4IM05 challenge. This document will explain how the project is divided and how to install every dependencies required to run the project properly.

### Project Distribution
This project has 4 main files:

- The first one, the data file, is not availabel in github and contains the data used during the project. It is a folder with two folders with the Train and Test images and their respective masks, and two .csv files that contains the metadata of each patient.
- The second folder is the research. This folder contains all the code created testing and implementing ideas that later on could have been used. It a folder containing garbage code that was used to experiment and try before going to the actual code.
- The third folder is the densenet. This Folder contains all the elements required to build the DenseNet, every component was done entirely by me with little to no help from LLM's. All this components are then ensemble together in the densente.py file that defines the network. 
- Finally the forth folder contains the runs, output of the profiler. It can be accessed by executing: `tensorboard --logdir=runs`. And then going on the broswer to the link: http://localhost:6006.

There are multiple files that are worth mentioning:

- densnet_trainer.py: This file defines a class that takes care of the whole pipeline required to train the DenseNet. It also handles the computation of metrics and calls the profiling.
- feature_extractor.py: This file defines a class that takes care of extracting the relevant features from the masks.
- model_weights.pth: This file contain the weights of the DenseNet, as well as in what epochs they were extracted and the state of the optimizer.
- niidataloader.py: This is the file that defined the dataset that handles the .nii MRI files. It also returns the voxel dimensions of the image.
- profiler.py: This file defines the class that handles the profiler. Given the input, this class creates the appropriate structure and saves it into the object to be uploaded to the runs folder.
- requirements.txt: This file contains the versions of the python packages used.
- roi.py: This file defines the class that implements the Region Of Interest (ROI) extraction.

### Presentation
The project's presentation, where the classification is done and some relevant topics are discussed, can be found in the presentation.ipynb file. This file was already run, but if you want to verify the outputs you can run it as well. Before running anything you should install the requirements in the requirements.txt, go to the Installation section to see how.
**Note:** I do not recommend running the presentation.ipynb file without a GPU, otherwise the computations could take quite a long time.


### Installation
To install all the dependencies this project requires first create a virtual environment. This can be done executing the following code:

`python -m venv .venv`

and enter it:

`source .venv/bin/activate`


Then check the torch website to determine which pytorch version is the adecuate for your computer: https://pytorch.org/get-started/locally/

Finally, run the following code to install all the dependencies of this project:

`pip install -r requirements.txt`