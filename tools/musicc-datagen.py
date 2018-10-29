import argparse
import configparser
import glob
import librosa
import pickle
import queue
import os
import threading
import sys


def fix_kwargs(kwargs):
    for key in kwargs:
        val = kwargs[key]
        try:
            new_val = int(val)
        except ValueError:
            try:
                new_val = float(val)
            except ValueError:
                if new_val in ('yes', 'true'):
                    new_val = True
                elif new_val in ('no', 'false'):
                    new_val = False
                else:
                    new_val = val
        kwargs[key] = new_val
    return kwargs

def extract_feature_from_config(y, sr, config, feature):
    kwargs = dict(config.items(feature))
    kwargs = fix_kwargs(kwargs)
    feature_func = getattr(librosa.feature, feature)
    return feature_func(y=y, sr=sr, **kwargs)

def extract_features_from_config(y, sr, config):
    for feature in config.sections():
        result = extract_feature_from_config(y, sr, config, feature)
        yield (feature, result)

def process_audio_file(audio_file, args):
    extra_kwargs = {}
    if args.duration is not None:
        extra_args["duration"] = args.duration
    if args.offset is not None:
        extra_kwargs["offset"] = args.offset
    y, sr = librosa.load(audio_file, **extra_kwargs)
    return y, sr

def worker_extractor(worker_queue, features, args):
    while True:
        music_path, entry = worker_queue.get()
        print("Start processing:", music_path)

        y, sr = process_audio_file(music_path, args)
        if args.extract_time_series:
            entry["y"] = y
            entry["sampling-rate"] = sr

        if args.extract_features:
            entry["features"] =\
                dict(extract_features_from_config(y, sr, config))
        print("Finish processing:", music_path)
        worker_queue.task_done()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="Path to the folder containing the dataset. "
                             "This directory should contain folders like "
                             "'rock', 'salsa', etc. and each folder should "
                             "contain audio files",
                        required=True)
    parser.add_argument("-e", "--extension",
                        help="Extension of the audio files",
                        required=True)
    parser.add_argument("-c", "--configfile",
                        type=argparse.FileType('r', encoding='utf-8'),
                        help="Path to the config file with features to use",
                        required=False)
    parser.add_argument("-d", "--extract-features", action="store_true",
                        required=False)
    parser.add_argument("-s", "--extract-time-series", action="store_true",
                        required=False)
    parser.add_argument("--duration", type=int, required=False)
    parser.add_argument("--offset", type=int, required=False)
    parser.add_argument("-o", "--output", type=argparse.FileType("wb"),
                        required=False)
    parser.add_argument("-t", "--worker-threads", type=int, required=False)

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read_file(args.configfile)

    features = {}

    worker_queue = queue.Queue()
    threads = []
    n_worker_threads = args.worker_threads or 25

    for i_genre, dirname in enumerate(os.listdir(args.path)):
        genre_path = os.path.join(args.path, dirname)
        audio_files = glob.glob(os.path.join(genre_path,
                                             "*.%s" % args.extension))

        for filename in audio_files:
            music_path = os.path.join(genre_path, filename)

            if not dirname in features:
                features[dirname] = {}

            features[dirname][music_path] = {
                "y": None,
                "sampling-rate": None,
                "features": {}
            }

            entry = (music_path, features[dirname][music_path])
            worker_queue.put(entry)

    for i in range(n_worker_threads):
        t = threading.Thread(target=worker_extractor,
                             args=(worker_queue, features, args))
        t.daemon = True
        t.start()
    worker_queue.join()

    if args.output:
        pickle.dump(features, args.output,
                    protocol=pickle.HIGHEST_PROTOCOL)
        args.output.close()
    else:
        print(features)
