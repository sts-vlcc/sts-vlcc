import torch as th
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import pandas as pd
import pickle
import math


class MC_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        subtitles_path,
        features_path,
        speaking_turns_path,
        use_speaking_turns_sampling=False,
        use_gibberish_subs=False,
        max_feats=10,
        features_dim=768,
        tokenizer=None,
        use_context=True,
        type_map=None,
        prefix="",
        suffix="",
    ):
        self.data = pd.read_csv(csv_path)
        if subtitles_path:
            self.subs = pickle.load(open(subtitles_path, "rb"))
        else:
            self.subs = None
        if speaking_turns_path:
            self.speaking_turns = pickle.load(open(speaking_turns_path, "rb"))
        else:
            self.speaking_turns = None
        self.use_gibberish_subs = use_gibberish_subs
        self.use_speaking_turns_sampling = use_speaking_turns_sampling
        print(f'use_speaking_turns_sampling: {use_speaking_turns_sampling} from {speaking_turns_path}')
        self.features = th.load(features_path)
        self.max_feats = max_feats
        self.features_dim = features_dim
        self.mask = tokenizer.mask_token if tokenizer is not None else None
        self.use_context = use_context
        mc = 0
        while f"a{mc}" in self.data:
            mc += 1
        self.mc = mc
        self.type_map = type_map
        self.prefix = prefix
        self.suffix = suffix

    def __len__(self):
        return len(self.data)

    def _get_subtitles(self, video_id, start, end):
        # only consider subtitles that intersec with the timestamps of the video clip
        subs_list = [
            x["text"]
            for x in self.subs[video_id]
            if x["end"] >= start and x["start"] <= end
        ]
        return " ".join(subs_list).capitalize().strip()

    def _get_text(self, subtitles, answer, mask, question=None):
        text = (
            f"{self.prefix} Question: {question} Is it '{answer}'? {mask}{self.suffix}"
        )
        if self.use_context:
            text += f" Subtitles: {subtitles}"
        text = text.strip()
        return text

    def _get_video(self, video_id, start, end):
        if video_id not in self.features:
            video = th.zeros(1, self.features_dim)
        else:
            if start is not None and not math.isnan(start):
                video = self.features[video_id][int(start) : int(end) + 1].float()
            else:
                video = self.features[video_id].float()
            if not len(video):
                print(video_id, start, end)
                video = th.zeros(1, self.features_dim)
        if self.speaking_turns is not None:
            if self.use_speaking_turns_sampling==True:
                
                # Retrieve and filter speaking turns for the video_id within start and end
                all_speaking_turns = self.speaking_turns.get(video_id, [])
                # speaking_turns = [turn for turn in all_speaking_turns if float(turn[0]) >= start and float(turn[1]) <= end]

                # Crop speaking turns to the video segment between start and end
                cropped_speaking_turns = []
                for turn in all_speaking_turns:
                    turn_start, turn_end = max(float(turn[0]), start), min(float(turn[1]), end)
                    # Only add the turn if there is an overlap with the video segment
                    if turn_start < turn_end:
                        cropped_speaking_turns.append([turn_start, turn_end] + turn[2:])


                # Convert string times to floats and calculate durations
                turns_with_durations = [(turn_start, turn_end, turn_end - turn_start, speaker)
                                        for turn_start, turn_end, speaker in cropped_speaking_turns]

                # Sort by duration to prioritize longer speaking turns
                turns_with_durations.sort(key=lambda x: x[2], reverse=True)

                if len(turns_with_durations) > self.max_feats:
                    turns_with_durations = turns_with_durations[:self.max_feats]

                # Restore original order
                turns_with_durations.sort(key=lambda x: cropped_speaking_turns.index([x[0], x[1], x[3]]))
                video_segments = []
                total_frames = 0
                
                for start, end, _,_ in turns_with_durations:
                    segment_len = int(end) - int(start) + 1
                    total_frames += segment_len
                    video_segments.append(self.features[video_id][int(start):int(end) + 1].float())
                # print(f"total_frames: {total_frames}")
                if total_frames > self.max_feats:
                    sampled_video = []
                    remaining_frames = self.max_feats
                    for segment in video_segments:
                        segment_len = len(segment)
                        # Calculate frames to sample, weighted by segment length
                        if remaining_frames > 1:  # Avoid division by zero in the last segment
                            frames_to_sample = max(1, round(segment_len / total_frames * remaining_frames))
                        else:
                            frames_to_sample = remaining_frames  # Assign any remaining frames to the last segment

                        # Ensure we do not sample more frames than are available in the segment
                        frames_to_sample = min(frames_to_sample, segment_len)

                        sampled = [segment[(j * segment_len) // frames_to_sample] for j in range(frames_to_sample)]
                        sampled_video.extend(sampled)

                        # Update the count of total and remaining frames
                        total_frames -= segment_len
                        remaining_frames -= frames_to_sample
                    video = th.stack(sampled_video)
                    video_len = self.max_feats
                else:
                    # print("total_frames <= self.max_feats")
                    if len(video) > self.max_feats:
                        sampled = []
                        for j in range(self.max_feats):
                            sampled.append(video[(j * len(video)) // self.max_feats])
                        video = th.stack(sampled)
                        video_len = self.max_feats
                    elif len(video) < self.max_feats:
                        video_len = len(video)
                        video = th.cat(
                            [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
                        )
                    else:
                        video_len = self.max_feats
            else:
                if len(video) > self.max_feats:
                    sampled = []
                    for j in range(self.max_feats):
                        sampled.append(video[(j * len(video)) // self.max_feats])
                    video = th.stack(sampled)
                    video_len = self.max_feats
                elif len(video) < self.max_feats:
                    video_len = len(video)
                    video = th.cat(
                        [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
                    )
                else:
                    video_len = self.max_feats
        else:
            if len(video) > self.max_feats:
                sampled = []
                for j in range(self.max_feats):
                    sampled.append(video[(j * len(video)) // self.max_feats])
                video = th.stack(sampled)
                video_len = self.max_feats
            elif len(video) < self.max_feats:
                video_len = len(video)
                video = th.cat(
                    [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
                )
            else:
                video_len = self.max_feats
        return video, video_len

    def __getitem__(self, idx):
        video_id = self.data["video_id"].values[idx]

        # get start, end
        start = self.data["start"].values[idx]
        end = self.data["end"].values[idx]

        # get question
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
        type = 0
        if "type" in self.data:
            type = self.data["type"].values[idx]

        # get subs
        if self.subs:
            if self.use_gibberish_subs:
                subs = "Gibberish"
            else:
                subs = self._get_subtitles(video_id, start, end)
            # print(f'subs: {subs}')
        else:
            subs = ""

        # get features
        video, video_len = self._get_video(video_id, start, end)

        # get answer id
        answer_id = -1  # for hidden set testing
        if "answer_id" in self.data:
            answer_id = self.data["answer_id"].values[idx]

        text = []
        for i in range(self.mc):
            ai = self.data[f"a{i}"].values[idx].capitalize().strip()
            text.append(self._get_text(subs, ai, self.mask, question))

        qid = idx
        if "qid" in self.data:
            qid = int(self.data["qid"].values[idx])

        return {
            "video": video,
            "video_len": video_len,
            "text": text,
            "qid": qid,
            "answer_id": answer_id,
            "type": type,
        }


def mc_collate_fn(batch):
    bs = len(batch)
    video = th.stack([batch[i]["video"] for i in range(bs)])
    video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
    text = [
        [batch[i]["text"][j] for i in range(bs)] for j in range(len(batch[0]["text"]))
    ]
    qid = [batch[i]["qid"] for i in range(bs)]
    answer_id = default_collate([batch[i]["answer_id"] for i in range(bs)])
    type = [batch[i]["type"] for i in range(bs)]

    return {
        "video": video,
        "video_len": video_len,
        "text": text,
        "qid": qid,
        "answer_id": answer_id,
        "type": type,
    }


def build_mc_dataset(dataset_name, split, args, tokenizer):
    type_map = None
    if dataset_name == "how2qa":
        if split == "train":
            csv_path = args.how2qa_train_csv_path
        elif split == "val":
            csv_path = args.how2qa_val_csv_path
        elif split == "test":
            csv_path = args.how2qa_val_csv_path  # eval on val public
        else:
            raise NotImplementedError
        subtitles_path = args.how2qa_subtitles_path
        features_path = args.how2qa_features_path
    elif dataset_name == "tvqa":
        if split == "train":
            csv_path = args.tvqa_train_csv_path
        elif split == "val":
            csv_path = args.tvqa_val_csv_path
        elif split == "test":
            csv_path = args.tvqa_test_csv_path
        else:
            raise NotImplementedError
        subtitles_path = args.tvqa_subtitles_path
        features_path = args.tvqa_features_path
    elif dataset_name == "siq2":
        if split == "train":
            csv_path = args.siq_train_csv_path
        elif split == "val":
            csv_path = args.siq_val_csv_path
        elif split == "test":
            csv_path = args.siq_test_csv_path
        else:
            raise NotImplementedError
        subtitles_path = args.siq_subtitles_path
        features_path = args.siq_features_path
        speaking_turns_path = args.siq_speaking_turns_path
    else:
        raise NotImplementedError
    return MC_Dataset(
        csv_path=csv_path,
        subtitles_path=subtitles_path,
        features_path=features_path,
        max_feats=args.max_feats,
        features_dim=args.features_dim,
        tokenizer=tokenizer,
        use_context=args.use_context,
        prefix=args.prefix,
        suffix=args.suffix,
        type_map=type_map,
        use_speaking_turns_sampling=args.use_speaking_turns_sampling,
        use_gibberish_subs=args.use_gibberish_subs,
        speaking_turns_path=speaking_turns_path if dataset_name == "siq2" else None,
    )
