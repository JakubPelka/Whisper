import os
import shutil

# Define the main directory and target subfolder
main_directory = r"C:\Users\jakpel\OneDrive - Kungsbacka kommun\Dokument\Ljudinspelningar"
target_subfolder = os.path.join(main_directory, "arkiv")

# Ensure the "arkiv" subfolder exists
if not os.path.exists(target_subfolder):
    print(f"Target subfolder '{target_subfolder}' does not exist. Please create it first.")
    exit()

# List of file extensions to look for
file_extensions = ['.m4a', '.mp3']

# Iterate through files in the main directory
for file_name in os.listdir(main_directory):
    file_path = os.path.join(main_directory, file_name)
    
    # Check if it's a file and has the correct extension
    if os.path.isfile(file_path) and any(file_name.lower().endswith(ext) for ext in file_extensions):
        # Move the file to the "arkiv" subfolder
        shutil.move(file_path, target_subfolder)
        print(f"Moved: {file_name}")

print("All matching files have been moved.")
