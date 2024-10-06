import os
import csv


def create_vid_paths_csv(vid_base_path, feature_out_base_path):
    # Create a csv file with the following columns : [video_path, feature_path]
    # An example entry in the csv file is as follows : [/home/admin-guest/Documents/multimodal-ml/Social_IQ_2_Data/siq2_clean/video/vTTzWRdAN4M.mp4, /home/admin-guest/Documents/multimodal-ml/FrozenBiLM/datasets/siq2/vid_features/vTTzWRdAN4M.npy]
    # This script is used to generate the csv file for the SIQ2 dataset by reading from a location and populating the video_path
    # The feature_path is the output folder path

    # Create the output folder if it does not exist
    if not os.path.exists(feature_out_base_path):
        os.makedirs(feature_out_base_path)

    # Create the csv file
    csv_file_path = os.path.join(feature_out_base_path, 'vid_defaced_transcript_paths.csv')
    with open(csv_file_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['video_path', 'feature_path'])

        # Iterate over all the videos in the video base path
        for root, dirs, files in os.walk(vid_base_path):
            for file in files:
                if file.endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    feature_path = os.path.join(feature_out_base_path, file.split('.')[0] + '.npy')
                    csv_writer.writerow([video_path, feature_path])



if __name__ == '__main__':
    # create_vid_paths_csv("/home/admin-guest/Documents/multimodal-ml/Social_IQ_2_Data/siq2_clean/video/",
                 #                      "/
    create_vid_paths_csv("/home/admin-guest/Documents/multimodal-ml/Social_IQ_2_Data/siq2/video_defaced",
                 "/home/admin-guest/Documents/multimodal-ml/FrozenBiLM_clip/datasets/SIQ2/new_feats/vid_defaced_contextualized_features")
