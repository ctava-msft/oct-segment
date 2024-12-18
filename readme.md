# Setup environment

Run the following commands to setup a python virtual env.

```
python -m venv .venv
pip install virtualenv
.venv\Scripts\activate
[linux]source .venv/bin/activate
pip install -r requirements.txt
```

Run the following command to download the SAM model on Linux:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Run the following command to download the SAM model on Windows:

```powershell
Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -OutFile sam_vit_h_4b8939.pth
```

To execute the script, type the following command:

python automatic_mask_generator.py <your_path_to_the_image>

### SAM2

SAM2 is an enhanced version of the Segment Anything Model. For more details, refer to the [SAM2 paper](https://arxiv.org/pdf/2408.00714).

## Design

### Coordinates to be able to draw a polygon.
layer_1_top = []
layer_1_bottom = []

# New Feature: Layer Coordinate Extraction

SAM can now extract gradient-based layer coordinates from OCT images. After mask prediction, top and bottom coordinates for each layer are saved to separate files (`layer_N_top.txt` and `layer_N_bottom.txt`).

## Usage

After running the mask prediction, check the project directory for the generated layer coordinate files.

```bash
python automatic_mask_generator.py <your_path_to_the_image>
```

The coordinates for each layer will be saved as `layer_1_top.txt`, `layer_1_bottom.txt`, etc.

Hey Guys - 

Good day. 

Have realized the SAM model was detecting objects without provided detailed segmentation coordinates.
That said, we were all attracted to the library for its ability to color code differences in the image gradients.
As such, it had promise to give us what we need. 
I was able to tweak the segment_anything source code to determine layers described as a coordinate system.

This was accomplished by...

Two options:
option one: use signal processing to help find edges and edge detection to get contours then and feed layers into SAM
option two: hack SAM - it knows about the gradients with colors, generatecoordinates from layers.

Thanks for your patience with the discovery process as I recognize this was the third attempt.

Hope you like it.

Here is a link to the repository with the code:

Here is the sample coordinate system:

Please provide any feedback you have.

Sincerely,
Chris