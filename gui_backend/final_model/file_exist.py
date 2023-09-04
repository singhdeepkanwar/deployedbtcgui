import os

file_path = '/home/ubuntu/MRI_classification_WebGUI/gui_backend/final_model/effnet.h5'

if os.path.isfile(file_path):
    print("File exists")
else:
    print("File does not exist")
