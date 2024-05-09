import os

def rename_images(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Sort the files alphabetically to ensure consistent renaming
    files.sort()
    
    # Counter for renaming
    count = 1
    
    # Rename each file
    for file in files:
        # Check if it's a file and not a directory
        if os.path.isfile(os.path.join(folder_path, file)):
            # Get the file extension
            _, extension = os.path.splitext(file)
            # Create the new file name
            new_name = str(count) + extension
            # Construct the full paths
            old_path = os.path.join(folder_path, file)
            new_path = os.path.join(folder_path, new_name[:-1])
            # Rename the file
            os.rename(old_path, new_path)
            count += 1

# Replace 'folder_path' with the path to your folder containing images
folder_path = ".\\char_sample\\"

for i in range(1,30):
    rename_images(folder_path + str(i) + "\\")