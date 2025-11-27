import os
import subprocess
import shutil
import re

def generate_mask(json_file):
    base_dir = os.getcwd()
    training_data_dir = os.path.join(base_dir, "training_data")
    json_path = os.path.join(training_data_dir, json_file)
    
    match = re.search(r'post(\d+)\.json', json_file)
    if not match:
        print("error: could not extract number from JSON filename")
        return
    
    file_number = match.group(1)
    
    original_cwd = os.getcwd()
    os.chdir(training_data_dir)
    
    try:
        result = subprocess.run(["labelme_json_to_dataset", json_file], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"error running labelme_json_to_dataset: {result.stderr}")
            return
        
        output_folder = json_file.replace('.json', '_json')
        output_folder_path = os.path.join(training_data_dir, output_folder)
        
        if not os.path.exists(output_folder_path):
            print(f"error: output folder {output_folder} was not created")
            return
        
        original_json_path = os.path.join(training_data_dir, json_file)
        destination_json_path = os.path.join(output_folder_path, json_file)
        
        if os.path.exists(original_json_path):
            shutil.move(original_json_path, destination_json_path)
        
        label_file_path = os.path.join(output_folder_path, "label.png")
        
        if os.path.exists(label_file_path):
            new_mask_name = f"mask{file_number}.png"
            new_mask_path = os.path.join(training_data_dir, new_mask_name)
            
            shutil.move(label_file_path, new_mask_path)
        else:
            print("warning: label.png not found in output folder")
            
    except Exception as e:
        print(f"error during processing: {e}")
    finally:
        os.chdir(original_cwd)

# example usage:
# generate_mask("post001.json")

generate_mask("post001.json")