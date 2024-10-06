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

filename = "datasets/SIQ2/transcripts/vtt/_0at8kXKWSw.vtt"
result = parse_vtt(filename)
for entry in result:
    # print(entry['start'], "-->", entry['end'])
    print(entry['text'])
    # print()