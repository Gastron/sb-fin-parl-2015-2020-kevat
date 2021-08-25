#!/usr/bin/env python3
"""Write Kaldi data as WebDataset shards"""

import webdataset as wds
import multiprocessing as mp
import subprocess
import torchaudio
import queue
import os
import pathlib
import warnings
import shutil
import kaldi_io
import torch


def kaldi_map_stream(path):
    with open(path) as fi:
        for line in fi:
            try:
                uttid, data = line.strip().split(maxsplit=1)
            except ValueError:
                # Empty entry
                uttid = line.strip().split(maxsplit=1)
                data = ""
            yield uttid, data

def read_rxwav(data):
    if data.endswith("|"):
        with subprocess.Popen(data, shell=True, stdout=subprocess.PIPE) as proc:
            signal, samplerate = torchaudio.load(proc.stdout, channels_first=False)
    else:
        signal, samplerate = torchaudio.load(data, channels_first=False)
    return signal, samplerate

def segments_to_output(segments_path, wavscp_path, fade_len=0.005):
    wavscp = dict(kaldi_map_stream(wavscp_path))
    current_id = None
    current_wav = None,
    current_samplerate = None
    current_fader = None
    for uttid, segment_data in kaldi_map_stream(segments_path):
        wav_id, start, end = segment_data.split()
        if wav_id != current_id:
            current_wav, current_samplerate = read_rxwav(
                    wavscp[wav_id]
            )
            current_id = wav_id
            fade_time = fade_len*current_samplerate
            current_fader = torchaudio.transforms.Fade(
                    fade_in_len=fade_time, 
                    fade_out_len=fade_time, 
            )
        start_ind = int(float(start) * current_samplerate)
        end_ind = int(float(end) * current_samplerate)
        output = {"audio.pth": current_fader(current_wav[start_ind:end_ind]),
                "meta.json": {"samplerate": current_samplerate}}
        yield uttid, output 

def wavscp_to_output(wavscp_path):
    for uttid, wavdata in kaldi_map_stream(wavscp_path):
        signal, samplerate = read_rxwav(wavdata)
        output = {"audio.pth": signal,
                "meta.json": {"samplerate": samplerate}}
        yield uttid, output

def text_to_output(text_path):
    for uttid, data in kaldi_map_stream(text_path):
        output = {"transcript.txt": data}
        yield uttid, output

def utt2spk_to_output(utt2spk_path):
    for uttid, data in kaldi_map_stream(utt2spk_path):
        output = {"meta.json": {"spkid":data}}
        yield uttid, output

def featsscp_to_output(featsscp_path):
    for uttid, feats in kaldi_io.read_mat_scp(featsscp_path):
        output = {"feats.pth": torch.from_numpy(feats)}
        yield uttid, output


def make_data_point(outputs):
    data_point = {}
    for uttid, output in outputs:
        if "__key__" not in data_point:
            data_point["__key__"] = uttid
        elif uttid != data_point["__key__"]:
            MSG = "Mismatched key, data probably not "
            MSG += "sorted and filtered the same way! "
            MSG += "Conflict: {uttid} != {data_point['__key__']}; "
            MSG += f"{' '.join(output.keys())} did not "
            MSG += f"match with {' '.join(data_point.keys())}"
            raise RuntimeError(MSG)
        for key, data in output.items():
            if isinstance(data, dict):
                to_update = data_point.setdefault(key, {})
                to_update.update(data)
            else:
                data_point[key] = data
    return data_point


STREAM_FUNCS = {
        "text": text_to_output,
        "segments": segments_to_output,
        "wavscp": wavscp_to_output,
        "utt2spk": utt2spk_to_output,
        "featsscp": featsscp_to_output
}
def make_streams(sources):
    streams = []
    for name, args in sources.items():
        stream = STREAM_FUNCS[name](*args)
        streams.append(stream)
    return streams
    

def write_shards(shard_dir, source_queue):
    shard_dir = pathlib.Path(shard_dir)
    shard_dir.mkdir(parents=True)
    shardpattern = f"{shard_dir}/shard-%06d.tar"
    with wds.ShardWriter(shardpattern, maxcount=500) as fo:
        while True:
            try:
                sources = source_queue.get(False)
            except queue.Empty:
                break
            streams = make_streams(sources)
            for outputs in zip(*streams):
                data_point = make_data_point(outputs)
                fo.write(data_point)

def fill_queue(split_dir, feats_mode=False):
    source_queue = mp.Queue()
    root, dirs, files = next(os.walk(split_dir))
    root = pathlib.Path(root)
    for split in dirs:
        splitpath = root / split
        sources = {}
        if not feats_mode:
            #segments / wav.scp
            if (splitpath / "segments").exists():
                if not (splitpath / "wav.scp").exists():
                    raise ValueError(f"{splitpath} has segments but not wav.scp")
                sources["segments"] = (splitpath / "segments", splitpath / "wav.scp")
            elif (splitpath / "wav.scp").exists():
                sources["wavscp"] = (splitpath / "wav.scp",)
            else:
                warnings.warn(f"No wav.scp nor segments file in {splitpath}")
        else:
            if not (splitpath / "feats.scp").exists():
                raise ValueError("Requested to use feats.scp but none found!")
            sources["featsscp"] = (str(splitpath / "feats.scp"),)
        #utt2spk
        if (splitpath / "utt2spk").exists():
            sources["utt2spk"] = (splitpath / "utt2spk",)
        else:
            warnings.warn(f"No utt2spk file in {splitpath}")
        #text
        if (splitpath / "text").exists():
            sources["text"] = (splitpath / "text",)
        else:
            warnings.warn(f"No text file in {splitpath}")
        source_queue.put(sources)
    return source_queue

def process_queue_in_parallel(source_queue, num_proc, shard_dir):
    shard_dir = pathlib.Path(shard_dir)
    processes = []
    for i in range(num_proc):
        proc = mp.Process(
                target=write_shards,
                kwargs={
                    "shard_dir": shard_dir / str(i),
                    "source_queue": source_queue
                }
        )
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()

def collect_shards(split_shard_dir, target_dir):
    shard_split_dir = pathlib.Path(split_shard_dir)
    target_dir = pathlib.Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    for shard_dir in shard_split_dir.iterdir():
        for shard in shard_dir.iterdir():
            shutil.move(shard, target_dir / ("shard-%06d.tar" % i))
            i += 1
        shard_dir.rmdir()
    shard_split_dir.rmdir()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir", 
            help="""A Kaldi top-level split dir, 
            with only data directories as sub-dirs""")
    parser.add_argument("shard_dir",
            help="""The top-level directory where 
            the shards should go.""",
            type=pathlib.Path)
    parser.add_argument("--num-proc",
            help="""The number of processes to use""",
            default=2,
            type=int)
    parser.add_argument("--feats-mode",
            help="""Use Kaldi features instead of
            raw audio""",
            action="store_true")
    args = parser.parse_args()
    source_queue = fill_queue(args.split_dir, feats_mode=args.feats_mode)
    process_queue_in_parallel(source_queue, args.num_proc, args.shard_dir / "TMP")
    collect_shards(args.shard_dir / "TMP", args.shard_dir) 

