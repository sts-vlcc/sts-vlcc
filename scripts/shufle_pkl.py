import pickle
import os
import random

def shuffle_and_merge_texts(input_folder, output_folder, output_merged_pkl):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_files = [f for f in os.listdir(input_folder) if f.endswith('.pkl')]
    merged_dict = {}
    
    for file in all_files:
        with open(os.path.join(input_folder, file), 'rb') as f:
            data = pickle.load(f)
            
            # Extract and shuffle the 'text' values
            original_texts = [entry['text'] for entry in data]
            shuffled_texts = original_texts.copy()
            random.shuffle(shuffled_texts)
            
            for entry in data:
                entry['text'] = shuffled_texts.pop()
            
            # Save the shuffled data to a new pkl in the output folder
            with open(os.path.join(output_folder, file), 'wb') as out_f:
                pickle.dump(data, out_f)
            
            # Add to merged dictionary
            file_key = file.replace('.pkl', '')
            merged_dict[file_key] = data  # Here we add the full list of dictionaries
    
    # Save the merged dictionary
    with open(output_merged_pkl, 'wb') as f:
        pickle.dump(merged_dict, f)


# Call the function with appropriate paths
shuffle_and_merge_texts('/home/admin-guest/Documents/multimodal-ml/FrozenBiLM/datasets/SIQ2/transcripts_shuffled/pkl',
              '/home/admin-guest/Documents/multimodal-ml/FrozenBiLM/datasets/SIQ2/transcripts_shuffled/pkl_trimmed', 
              '/home/admin-guest/Documents/multimodal-ml/FrozenBiLM/datasets/SIQ2/subtitles_shuffled_trimmed_2.pkl')
