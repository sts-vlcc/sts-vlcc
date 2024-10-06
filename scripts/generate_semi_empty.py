import os
import argparse

def semi_empty_vtt(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write('WEBVTT\n\n')  # Header for the vtt file
        
        for line in lines:
            line = line.strip()
            # Check if the line is a timestamp
            if '-->' in line:
                outfile.write(line + '\nGibberish\n')  # Write the timestamp with two newlines
                return  # Stop processing the file

def process_folder(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.vtt'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            semi_empty_vtt(input_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Generate semi-empty VTT files.")
    parser.add_argument('input_folder', type=str, help="Path to the input folder containing VTT files.")
    parser.add_argument('output_folder', type=str, help="Path to the output folder where the semi-empty VTT files will be saved.")
    
    args = parser.parse_args()
    
    process_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()