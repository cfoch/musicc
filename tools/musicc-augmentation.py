import argparse
import configparser
import glob
import librosa
import pickle
import numpy as np
import os
import threading
import sys


def process_audio_file(audio_file, args):
    extra_kwargs = {}
    if args.sampling_rate is not None:
        extra_kwargs["sr"] = args.sampling_rate
    y, sr = librosa.load(audio_file, **extra_kwargs)
    return y, sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="Path to the folder containing the dataset. "
                             "This directory should contain folders like "
                             "'rock', 'salsa', etc. and each folder should "
                             "contain audio files",
                        required=True)
    parser.add_argument("-o", "--output-path",
                        help="Path to the folder containing the dataset with "
                             "the output data.",
                        required=True)
    parser.add_argument("-g", "--genre",
                        help="Gender to consider",
                        required=False)
    parser.add_argument("-e", "--extension",
                        help="Extension of the audio files",
                        required=True)
    parser.add_argument("--sampling-rate",
                        type=int,
                        default=22050,
                        required=False)
    parser.add_argument("--add-noise",
                        action="store_true",
                        required=False)
    parser.add_argument("--noise-min",
                        type=float,
                        default=0.005,
                        required=False)
    parser.add_argument("--noise-max",
                        type=float,
                        default=0.005,
                        required=False)
    parser.add_argument("--noise-step",
                        type=float,
                        default=0.005,
                        required=False)

    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        print("Error, '%s' does not exist or isn't a directory." %
              args.output_path, file=sys.stderr)
        sys.exit(1)

    for i_genre, dirname in enumerate(os.listdir(args.path)):
        genre_path = os.path.join(args.path, dirname)
        audio_files = glob.glob(os.path.join(genre_path,
                                             "*.%s" % args.extension))
        genre_path = os.path.join(args.path, dirname)
        audio_files = glob.glob(os.path.join(genre_path,
                                             "*.%s" % args.extension))

        for filename in audio_files:
            music_path = os.path.join(genre_path, filename)
            output_genre_path = os.path.join(args.output_path, dirname)
            basename, _ = os.path.splitext(os.path.basename(filename))
            output_music_path =\
                os.path.join(output_genre_path, "%s.wav" % basename)
            # Just make a copy of the file.
            try:
                os.makedirs(output_genre_path)
            except FileExistsError:
                pass
            print("Copying file '%s' resampled to %d to '%s'." %
                  (music_path, args.sampling_rate, output_music_path),
                  file=sys.stderr)

            y, sr = process_audio_file(music_path, args)
            librosa.output.write_wav(output_music_path, y, sr)

            if args.add_noise:
                noise_range = np.arange(args.noise_min,
                                        args.noise_max + args.noise_step,
                                        args.noise_step)
                for i, noise_factor in enumerate(noise_range):
                    output_filename, ext = os.path.splitext(os.path.basename(filename))
                    output_music_path =\
                        os.path.join(output_genre_path,
                                     "%s-noise-%d.wav" % (output_filename, i))

                    print("Adding noise to file '%s' and saving it in '%s'." %
                          (music_path, output_music_path), file=sys.stderr)
                    # Generate noise.
                    noise_amp = noise_factor * np.random.uniform() * np.amax(y)
                    y_with_noise = y.astype('float64')\
                        + noise_amp * np.random.normal(size=y.shape[0])
                    # Save file with noise.
                    librosa.output.write_wav(output_music_path,
                                             y_with_noise, sr)
