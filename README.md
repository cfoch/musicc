# musicc
Music is a library to get features and classify sound easily. It is based on librosa.

## Tools
### musicc-datagen
This is a CLI that allows you extract features from a bunch of audio files from a configuration file. The user is expected to have the files organized in this following way:
```
genres
├── blues
    ├── file1.au
    ├── file2au
    ...
├── classical
    ├── file1.au
    ├── file2au
    ...
├── country
    ├── file1.au
    ├── file2au
    ...
├── disco
    ├── file1.au
    ├── file2au
    ...
...
```
So for example if this folder is in `/home/foo/Downloads/`, you can generate a pickle with the extracted features by doing:
```
python3 tools/musicc-datagen.py -p /home/foo/Downloads/genres/ -e au -c features.cfg -d -o features.pickle
```
This command line will generate a `pickle` file with the extracted features of all the `.au` files located at `/home/foo/Downloads/genres/`.

The format of the `features.cfg` file is given here:
```
[librosa_spectral_feature_function_name]
argument1=value
argument2=value
argument3=value
[other_librosa_spectral_feature_function_name]
argument1=value
argument2=value
argument3=value
```
For example, the `features.cfg` file could be something like
```
[mfcc]
hop_length=512
n_mfcc=13
[spectral_centroid]
n_fft=1024
hop_length=512
```
To know which arguments to use, read more information about in the [librosa's documentation](https://librosa.github.io/librosa/feature.html). As you can see you just need to specify which features you'd like to extract in that configuration file. This will save you the time of writing code to do that!

Finally, with the generated pickle you can do whatever you want. For example, once the pickle is generated you can write a little Python program that does the following:
```
with open('filename.pickle', 'rb') as f:
    data = pickle.load(f)
    print(data)
```
