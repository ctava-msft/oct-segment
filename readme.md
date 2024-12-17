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