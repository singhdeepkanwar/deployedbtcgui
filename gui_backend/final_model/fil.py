import h5py

with h5py.File('/home/ubuntu/MRI_classification_WebGUI/gui_backend/final_model/effnet.h5', 'r') as file:
    # Perform any required operations 
    print("File opened successfully")
