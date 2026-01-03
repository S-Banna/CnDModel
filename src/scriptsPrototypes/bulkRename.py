import os

def rename_files():
    # temp folder used here, not tracked in git
    folder_path = 'temp'
    if not os.path.exists(folder_path):
        print(f"error: folder '{folder_path}' does not exist")
        return
    
    files = os.listdir(folder_path) 
    renamed_count = 0
    
    for filename in files:
        parts = filename.split('_')
        
        if len(parts) >= 3:
            # first split is the identifier
            number = parts[0]
            
            if len(number) == 3 and number.isdigit():
                # look for 'pre' or 'post' in the filename (case insensitive)
                lower_filename = filename.lower()
                
                if 'pre' in lower_filename:
                    specifier = 'pre'
                elif 'post' in lower_filename:
                    specifier = 'post'
                else:
                    continue
                
                # get file extension
                name, extension = os.path.splitext(filename)
                new_name = f"{number}{specifier}{extension}"
                
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_name)
                
                # check if new filename already exists
                if os.path.exists(new_path):
                    print(f"warning: {new_name} already exists, skipping {filename}")
                    continue
                
                # rename the file
                os.rename(old_path, new_path)
                print(f"renamed: {filename} -> {new_name}")
                renamed_count += 1
    
    print(f"\nrenaming complete! {renamed_count} files were renamed")

if __name__ == "__main__":
    rename_files()