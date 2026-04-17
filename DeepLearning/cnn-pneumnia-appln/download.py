import os
import zipfile
# download dataset
os.system("kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
# unzip
with zipfile.ZipFile("chest-xray-pneumonia.zip", 'r') as zip_ref:
    zip_ref.extractall("data")