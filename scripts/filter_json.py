import os
import json
import argparse

def is_vtt_empty(vtt_path):
    """Check if a VTT file is empty."""
    with open(vtt_path, 'r') as f:
        content = f.read().strip()
        # Check if the content only contains the "WEBVTT" header
        return content == "WEBVTT"

def filter_json_by_existing_videos(video_folder, transcript_folder, input_json_path, output_json_path):
    # Get a list of all .mp4 files in the video folder
    existing_videos = set([f.split('.')[0] for f in os.listdir(video_folder) if f.endswith('.mp4')])
    existing_transcripts = set([f.split('.')[0] for f in os.listdir(transcript_folder) if f.endswith('.vtt')])

    empty_vtts = []

    # Load multiple JSON objects from the file
    with open(input_json_path, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Get the unique vid_name values before filtering
    unique_vid_names_before = set([entry['vid_name'] for entry in data])

    # Filter out entries based on conditions
    filtered_data = []
    for entry in data:
        vid_name = entry["vid_name"]
        if vid_name not in existing_videos:
            continue
        if vid_name not in existing_transcripts:
            continue
        if is_vtt_empty(os.path.join(transcript_folder, vid_name + ".vtt")):
            empty_vtts.append(vid_name)
            continue
        filtered_data.append(entry)

    # Get the unique vid_name values after filtering
    unique_vid_names_after = set([entry['vid_name'] for entry in filtered_data])

    # Write the filtered data to the output JSON file
    with open(output_json_path, 'w') as f:
        for entry in filtered_data:
            f.write(json.dumps(entry) + '\n')

    return len(unique_vid_names_before), len(unique_vid_names_after), empty_vtts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a JSON file based on existing .mp4 videos and non-empty VTT transcripts in folders.")
    parser.add_argument("video_folder", help="Path to the folder containing .mp4 videos.")
    parser.add_argument("transcript_folder", help="Path to the folder containing .vtt transcript files.")
    parser.add_argument("input_json_path", help="Path to the initial JSON file.")
    parser.add_argument("output_json_path", help="Destination path for the filtered JSON file.")
    
    args = parser.parse_args()

    unique_before, unique_after, empty_vtts = filter_json_by_existing_videos(args.video_folder, args.transcript_folder, args.input_json_path, args.output_json_path)
    print(f"Number of unique vid_name before filtering: {unique_before}")
    print(f"Number of unique vid_name after filtering: {unique_after}")
    print(f"Number of empty VTT files: {len(empty_vtts)}")
    print("Empty VTT filenames:", ", ".join(empty_vtts))