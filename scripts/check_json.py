import json
import os
import argparse

def check_files_in_folder(json_file, folder,format):
    # Read the JSON data
    with open(json_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Extract all unique vid_name values from the JSON data using a set for deduplication
    vid_names = set([item['vid_name'] for item in data])

    # Check for corresponding files in the folder
    missing_files = []
    for vid in vid_names:
        # Assuming you are looking for .mp3 files for the given vid_name
        filename = f"{vid}."+format
        if not os.path.exists(os.path.join(folder, filename)):
            missing_files.append(filename)

    # Print the results
    if not missing_files:
        print("All referenced vid_name have corresponding files in the folder.")
    else:
        print(f"Missing files for the following vid_name(s): {', '.join(missing_files)}")

if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser(description="Check if vid_name in JSON file has a corresponding file in folder.")
    parser.add_argument('json_file', type=str, help="Path to the JSON file.")
    parser.add_argument('folder', type=str, help="Path to the folder containing potential files.")
    parser.add_argument('format', type=str, help='Format of the elements to delete')

    args = parser.parse_args()

    check_files_in_folder(args.json_file, args.folder,args.format)