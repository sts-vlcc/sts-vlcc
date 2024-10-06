import json
import csv
import argparse

def json_to_csv(json_file, csv_filename):
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Prepare the CSV data
    csv_data = []
    for idx, item in enumerate(data):
        start, end = item['ts'].split('-')
        end = int(float(end))
        csv_data.append([
            idx,  # qid
            item['q'],  # question
            item['answer_idx'],  # answer_id
            item['vid_name'],  # video_id
            item['a0'],
            item['a1'],
            item['a2'],
            item['a3'],
            0,  # start
            end  # end
        ])

    # Write to CSV file
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['qid', 'question', 'answer_id', 'video_id', 'a0', 'a1', 'a2', 'a3', 'start', 'end'])
        writer.writerows(csv_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to CSV.")
    parser.add_argument("json_file", help="Path to the input JSON file.")
    parser.add_argument("csv_filename", help="Name of the output CSV file.")
    args = parser.parse_args()

    json_to_csv(args.json_file, args.csv_filename)
    print(f"Data has been written to {args.csv_filename} successfully!")