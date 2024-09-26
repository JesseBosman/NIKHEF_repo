import os

folder_path = "/data/antares/users/jwbosman/results/ORCA10 copy/track/2d_hists/"


# Get the list of files in the folder
files = os.listdir(folder_path)

# Iterate over each file
for file_name in files:
    # Create the new file name by replacing invalid characters
    new_file_name = file_name.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")

    # Rename the file
    os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))