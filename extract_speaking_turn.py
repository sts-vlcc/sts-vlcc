import torch as th
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from extract.video_loader import VideoLoader
from torch.utils.data import DataLoader
from extract.preprocessing import Preprocessing
from extract.random_sequence_shuffler import RandomSequenceSampler
from args import MODEL_DIR
import re
import clip
import pickle
parser = argparse.ArgumentParser(description="Easy video feature extractor")

parser.add_argument(
    "--csv",
    type=str,
    help="input csv with columns video_path (input video) and feature_path (output path to feature)",
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="batch size for extraction"
)
parser.add_argument(
    "--half_precision",
    type=int,
    default=1,
    help="whether to output half precision float or not",
)
parser.add_argument(
    "--num_decoding_thread",
    type=int,
    default=1,
    help="number of parallel threads for video decoding",
)
parser.add_argument(
    "--l2_normalize",
    type=int,
    default=0,
    help="whether to l2 normalize the output feature",
)
parser.add_argument(
    "--feature_dim", type=int, default=768*2, help="output video feature dimension"
)

parser.add_argument(
    "--frame_feature_dim", type=int, default=768, help="output video feature dimension"
)

parser.add_argument(
    "--text_feature_dim", type=int, default=768, help="output video feature dimension"
)


args = parser.parse_args()

dataset = VideoLoader(
    args.csv,
    framerate=1,  # one feature per second max
    size=224,
    centercrop=True,
)


n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    sampler=sampler if n_dataset > 10 else None,
)
preprocess = Preprocessing()
model, _ = clip.load("ViT-L/14", download_root=MODEL_DIR)
model.eval()
model = model.cuda()


def extract_video_id(path):
    # Split the path by '/' to get the individual components
    parts = path.split('/')
    # Get the last part which is assumed to be the file name
    file_name = parts[-1]
    # Split the file name by '.' to separate the name and extension
    name_parts = file_name.split('.')
    # The video id is assumed to be the part before the last dot
    video_id = name_parts[-2] if len(name_parts) > 1 else None
    return video_id
# Function to extract text features for each frame and concatenate with video features
def extract_and_concatenate_features(video_features, frame_texts, model, text_feature_dim, frame_feature_dim):
    # Ensure the video_features are the correct type and on the right device
    video_features = video_features.float().cuda()
    # Initialize a tensor to hold all text features
    # print(f'frame_texts {frame_texts}')
    text_features_list = th.cuda.FloatTensor(len(frame_texts), text_feature_dim).fill_(0).half()

    # Encode each text in the frame_texts list
    for i, text in enumerate(frame_texts):
        # self.x = clip.tokenize(text, context_length=77, truncate=True)
        text_tokens = clip.tokenize([text], context_length=77, truncate=True).cuda()
        text_features = model.encode_text(text_tokens)
        text_features = text_features.float()  # Ensure text features are in full precision
        # print(f'text_features shape = {text_features.shape}')
        text_features_list[i] = text_features[:, :text_feature_dim].half()  # Store the text features

    # If there are more video features than text features, pad the text features with zeros
    if video_features.shape[0] > text_features_list.shape[0]:
        padding = th.zeros((video_features.shape[0] - text_features_list.shape[0], text_feature_dim), device='cuda').half()
        text_features_list = th.cat((text_features_list, padding), dim=0)

    # If there are more text features than video features, pad the video features with zeros
    if video_features.shape[0] < text_features_list.shape[0]:
        padding = th.zeros((text_features_list.shape[0] - video_features.shape[0], frame_feature_dim), device='cuda').half()
        video_features = th.cat((video_features, padding), dim=0)
    # print(f'text features shape = {text_features_list.shape}')
    # Concatenate video and text features along the feature dimension
    concatenated_features = th.cat((video_features, text_features_list), dim=1)
    
    if concatenated_features.shape[-1] != 1536:
        print(f'error in concatenation, shape = {concatenated_features.shape}')
    else:
            print(f'concatenated features shape = {concatenated_features.shape}')
    return concatenated_features

# projection_layer = th.nn.Linear(768, args.frame_feature_dim)  # Projection from 768 to 512
# projection_layer = projection_layer.cuda()
# projection_layer = projection_layer.half()
# Dictionary of video IDs and associated texts
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

# Get speaking turn information for each video
data_speaking_turns = pickle.load(open('datasets/SIQ2/turn_information.pkl', "rb"))

data_transcripts_speaking_turns = {}
for video in data_speaking_turns:
    # /home/admin-guest/Documents/multimodal-ml/FrozenBiLM_clip/datasets/SIQ2/transcripts/vtt
    try :
        transcript = parse_vtt(f'/home/admin-guest/Documents/multimodal-ml/Social_IQ_2_Data/siq2_clean/transcript/{video}.vtt')
        # transcript = parse_vtt(f'/home/admin-guest/Documents/multimodal-ml/FrozenBiLM_clip/datasets/SIQ2/transcripts/vtt/{video}.vtt')
        data_transcripts_speaking_turns[video]=[]
        for turn in data_speaking_turns[video]:
            st_start = turn[0]
            st_end = turn[1]
            data_transcripts_speaking_turns[video].append([st_start,st_end,""])
            for line in transcript:
                if line['end']>=st_start and line['start']<=st_end:
                    data_transcripts_speaking_turns[video][-1][2] = " ".join([data_transcripts_speaking_turns[video][-1][2], line['text']])
    except:
        print(f'no transcript for {video}')
        pass

video_texts = {}

def fill_transcript_gaps(transcript):
    output = []
    current_second = 0
    
    for segment in transcript:
        # Unpack the list
        start, end, text = segment
        start, end = int(start), int(end)
        
        # Fill in the empty seconds with ' '
        for _ in range(start - current_second):
            output.append(' ')
        
        # Repeat the text for the duration it spans
        for _ in range(end - start + 1):
            output.append(text)
        
        # Update the current_second to the end of the last segment
        current_second = end + 1
    
    return output

for video in data_transcripts_speaking_turns:
    video_texts[video]=[]
    start=0
    # print(data_transcripts_speaking_turns[video])
    for st in data_transcripts_speaking_turns[video]:
        st_start = int(st[0])
        gap = int(st[1]-st[0])
        while start<st_start:
            video_texts[video].append("")
            start+=1
        for _ in range(gap):
            video_texts[video].append(st[2])
        start +=gap
    
with th.no_grad():
    for k, data in tqdm(enumerate(loader), total=len(loader)):
        input_file = data["input"][0]
        # print(input_file)
        output_file = data["output"][0]
        # import pdb; pdb.set_trace()
        video_id = extract_video_id(input_file)
        video = data["video"].squeeze()
        test_text = []
        # Retrieve the list of texts for the current video

        frame_texts = video_texts.get(video_id, [])

        if len(data["video"].shape) > 3:
            print(
                "Computing features of video {}/{}: {}".format(
                    k + 1, n_dataset, input_file
                )
            )
            video = data["video"].squeeze()
            if len(video.shape) == 4:
                video = preprocess(video)
                n_chunk = len(video)
                features = th.cuda.FloatTensor(n_chunk, args.frame_feature_dim).fill_(0)  # args.frame_feature_dim should be 512
                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                for i in range(n_iter):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    # print(f'min_ind = {min_ind}, max_ind = {max_ind}')
                    video_batch = video[min_ind:max_ind].cuda()
                    batch_features = model.encode_image(video_batch)
                    # print(f'batch features shape: {batch_features.shape}')
                    # batch_features = projection_layer(batch_features)
                    # print(f'batch features shape: {batch_features.shape}')
                    if args.l2_normalize:
                        batch_features = F.normalize(batch_features, dim=1)

                    features[min_ind:max_ind] = batch_features
                # print(f'features shape before = {features.shape}')
                # print(f'features shape before = {features.shape}')
                concatenated_features = extract_and_concatenate_features(features, frame_texts, model, args.text_feature_dim, args.frame_feature_dim)
                features = concatenated_features.cpu().numpy()
                # print(f'features shape after = {features.shape}')
                if args.half_precision:
                    pass
                np.save(output_file, features)
            # print(f'features shape = {features.shape}')
            ##ADD CODE TO EXTRACT TEXT FEATURES
        else:
            print("Video {} already processed.".format(input_file))
