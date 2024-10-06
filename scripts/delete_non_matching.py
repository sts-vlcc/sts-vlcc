import argparse
import os

def get_files_without_extension(directory):
    """Get a set of base filenames without their extensions from a directory."""
    return {os.path.splitext(f)[0] for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))}

def delete_non_matching_files(directory, extension, matching_names):
    """Delete files in the directory that don't have their base names in matching_names."""
    for f in os.listdir(directory):
        full_path = os.path.join(directory, f)
        base_name, ext = os.path.splitext(f)
        if os.path.isfile(full_path) and ext == extension and base_name not in matching_names:
            os.remove(full_path)

def main():
    parser = argparse.ArgumentParser(description='Keep only the intersection of the files across two folders.')
    parser.add_argument('dirA', type=str, help='Path to the first directory (e.g., contains .mp4 files)')
    parser.add_argument('dirB', type=str, help='Path to the second directory (e.g., contains .mp3, .wav files)')
    args = parser.parse_args()

    # Get the intersection of file base names between the two directories
    intersecting_names = get_files_without_extension(args.dirA).intersection(get_files_without_extension(args.dirB))

    # Delete non-matching files in dirA and dirB
    delete_non_matching_files(args.dirA, '.mp4', intersecting_names)  # Assuming dirA has .mp4 files
    for ext in ['.mp3', '.wav','.vtt']:  # Assuming dirB can have .mp3 and .wav files
        delete_non_matching_files(args.dirB, ext, intersecting_names)

if __name__ == '__main__':
    main()

