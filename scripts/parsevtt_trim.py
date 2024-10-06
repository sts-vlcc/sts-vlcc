import os
import re
import pickle
import argparse

def parse_vtt(filename):
    with open(filename, "r") as file:
        content = file.read()

    lines = content.strip().split("\n")
    lines = lines[1:]

    parsed_data = []
    prev_line = None

    i = 0
    while i < len(lines):
        text_lines = []

        # Check if the line matches a timestamp
        match = re.match(r"(\d+:\d+:\d+\.\d+) --> (\d+:\d+:\d+\.\d+)", lines[i])
        if match:
            start_time, end_time = match.groups()
            start_time = sum(float(x) * 60 ** idx for idx, x in enumerate(reversed(start_time.split(":"))))
            end_time = sum(float(x) * 60 ** idx for idx, x in enumerate(reversed(end_time.split(":"))))
            i += 1

            while i < len(lines) and not re.match(r"(\d+:\d+:\d+\.\d+) --> (\d+:\d+:\d+\.\d+)", lines[i]) and lines[i] != "":
                if lines[i] == ' ': # if the line is a space, skip it
                    # print("line empty",lines[i])
                    pass
                else:                    
                    if lines[i] != prev_line:
                        text_lines.append(lines[i])
                    prev_line = lines[i]
                    # print("line ",lines[i])
                i += 1

            current_text = " ".join(text_lines).strip()

            # If the current_text is not empty, add it to parsed_data
            if current_text:
                parsed_data.append({
                    'text': current_text,
                    'start': start_time,
                    'end': end_time
                })
        else:
            i += 1

    return parsed_data

def save_to_pkl(data, filename):
    with open(filename, "wb") as file:
        pickle.dump(data, file)

def process_folder(input_folder, output_folder):
    merged_data = {}
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".vtt"):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".pkl")

                parsed_data = parse_vtt(input_file_path)
                save_to_pkl(parsed_data, output_file_path)

                merged_data[os.path.splitext(file)[0]] = parsed_data

    return merged_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process VTT files and save them as PKL.")
    parser.add_argument("input_folder", help="Path to the input folder containing VTT files.")
    parser.add_argument("output_folder", help="Path to the output folder where PKL files will be saved.")
    parser.add_argument("merged_filename", help="Name for the merged PKL file (without extension).")

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    merged_data = process_folder(args.input_folder, args.output_folder)
    save_to_pkl(merged_data, args.merged_filename + ".pkl")