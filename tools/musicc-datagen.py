import argparse
import configparser
import glob
import librosa
import librosa.display
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import os
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

def process_audio_file(audio_file):
    y, sr = librosa.load(filename)
    return y, sr


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
    parser.add_argument("-o", "--output", type=argparse.FileType("wb"),
                        required=False)

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read_file(args.configfile)

    features = {}
    for dirname in os.listdir(args.path):
        genre_path = os.path.join(args.path, dirname)
        audio_files = glob.glob(os.path.join(genre_path,
                                             "*.%s" % args.extension))
        for filename in audio_files:
            music_path = os.path.join(genre_path, filename)
            y, sr = process_audio_file(music_path)

            print(filename)
            if not dirname in features:
                features[dirname] = {}
                # _, music_filename = os.path.split(music_path)
                print(music_path, file=sys.stderr)
                if not music_path in features:
                    features[dirname][music_path] = {
                        "y": y,
                        "sampling-rate": sr,
                        "features": []
                    }
                if args.extract_features:
                    features[dirname][music_path] =\
                        dict(extract_features_from_config(y, sr, config))
    if args.output:
        pickle.dump(features, args.output,
                    protocol=pickle.HIGHEST_PROTOCOL)
        args.output.close()
    else:
        print(features)
