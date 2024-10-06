import argparse
import json
import os

def get_referenced_vid_names_from_json(json_file_path):
    """
    Extract the 'vid_name' field from the given JSON file and return as a set.
    """
    with open(json_file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return set([entry['vid_name'] for entry in data])

def get_all_files(folder_path,format = "mp4"):
    """
    Get all VTT files from the specified folder and return as a set.
    """
    return set([filename[:-4] for filename in os.listdir(folder_path) if filename.endswith(format)])

def delete_unreferenced_files(unreferenced_files, folder_path,format):
    """
    Delete all unreferenced VTT files from the specified folder.
    """
    for vtt in unreferenced_files:
        vtt_path = os.path.join(folder_path, vtt + "." + format)
        try:
            os.remove(vtt_path)
            print(f"Deleted: {vtt_path}")
        except Exception as e:
            print(f"Error deleting {vtt_path}. Reason: {e}")

def main():
    parser = argparse.ArgumentParser(description='Find VTT files not referenced in JSON files.')
    parser.add_argument('json_file1', type=str, help='Path to the first JSON file.')
    parser.add_argument('json_file2', type=str, help='Path to the second JSON file.')
    parser.add_argument('json_file3', type=str, help='Path to the third JSON file.')
    parser.add_argument('vtt_folder', type=str, help='Path to the folder containing VTT files.')
    parser.add_argument('mp4_folder', type=str, help='Path to the folder containing mp4 files.')
    parser.add_argument('to_clean_folder', type=str, help='Path to the folder containing deletable files.')
    parser.add_argument('format', type=str, help='Format of the elements to delete')
    
    args = parser.parse_args()
    
    referenced_vid_names = set()
    referenced_vid_names.update(get_referenced_vid_names_from_json(args.json_file1))
    referenced_vid_names.update(get_referenced_vid_names_from_json(args.json_file2))
    referenced_vid_names.update(get_referenced_vid_names_from_json(args.json_file3))
    
    all_vtt_files = get_all_files(args.vtt_folder,".vtt")
    all_mp4_files = get_all_files(args.mp4_folder,".mp4")

    print(len(referenced_vid_names))
    print(len(all_vtt_files))
    print(len(all_mp4_files))

    unreferenced_vtt_files = all_vtt_files - referenced_vid_names
    unreferenced_mp4_files = all_mp4_files - referenced_vid_names

    print("Unreferenced VTT files:")
    # for vtt in unreferenced_vtt_files:
        # print(vtt)
    print(len(unreferenced_vtt_files))
    print("Unreferenced mp4 files:")

    print(len(unreferenced_mp4_files))
    delete_unreferenced_files(unreferenced_vtt_files, args.to_clean_folder,args.format)

    

if __name__ == '__main__':
    main()