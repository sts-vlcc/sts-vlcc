import csv
import argparse

def split_segments(start, end):
    # Calculate the segments
    half = (end - start) // 2
    quarter = (end - start) // 4
    return [
        (start, start + half),
        (start + half, end),
        (start + quarter, start + 3 * quarter)
    ]

def process_csv(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write the header
        header = next(reader)
        writer.writerow(header)

        for row in reader:
            start, end = int(row[8]), int(row[9])
            for new_start, new_end in split_segments(start, end):
                new_row = row.copy()
                new_row[8], new_row[9] = str(new_start), str(new_end)
                writer.writerow(new_row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CSV rows into 3 segments based on 'start' and 'end' columns.")
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("output_csv", help="Path to the output CSV file.")

    args = parser.parse_args()

    process_csv(args.input_csv, args.output_csv)