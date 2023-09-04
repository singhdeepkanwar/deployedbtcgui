import h5py

file_path = '/home/ubuntu/MRI_classification_WebGUI/gui_backend/final_model/effnet.h5'

try:
    with h5py.File(file_path, 'r') as file:
        # Perform any required operations on the file
        print("File opened successfully")
except OSError:
    print("Error opening the file")
