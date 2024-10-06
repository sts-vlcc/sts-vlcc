import pickle
import os
import webvtt
import re

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
data_speaking_turns = pickle.load(open('turn_information.pkl', "rb"))

data_transcripts_speaking_turns = {}
for video in data_speaking_turns:
    transcript = parse_vtt(f'/home/admin-guest/Documents/multimodal-ml/Social_IQ_2_Data/siq2_clean/transcript/{video}.vtt')
    data_transcripts_speaking_turns[video]=[]
    for turn in data_speaking_turns[video]:
        st_start = turn[0]
        st_end = turn[1]
        data_transcripts_speaking_turns[video].append([st_start,st_end,""])
        for line in transcript:
            if line['end']>=st_start and line['start']<=st_end:
                data_transcripts_speaking_turns[video][-1][2] = " ".join([data_transcripts_speaking_turns[video][-1][2], line['text']])

augmented = {}
for video in data_transcripts_speaking_turns:
    augmented[video]=[]
    start=0
    for st in data_transcripts_speaking_turns[video]:
        st_start = int(st[0])
        gap = int(st[1]-st[0])
        while start<st_start:
            augmented[video].append("")
            start+=1
        for _ in range(gap):
            augmented[video].append(st[2])
        start +=gap