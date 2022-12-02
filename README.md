# Speaker Identification Using Meta-Learning (Stanford CS330 Final Project by John Boccio)
This project aims to take a meta-learning approach to solving the speaker identification problem. The speaker 
identification problem consists of being to identify who is talking within a given audio clip from a given set of speakers.
A few examples of audio clips of where this user is talking is provided beforehand and used to train the model in a 
few-shot fashion.

## Dataset (VoxCeleb)
The dataset that is being used is to train the speaker identification model is 
[VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/). Since audio data can be very large, this project will be
utlizing the smaller VoxCeleb1 dataset which consists of 153,516 uterrances from 1,251 celebrities. This data was 
collected by scraping data off of youtube and was then labeled with timestamps of when each celebrity is talking in
those audio clips. The data is no longer available for easy download and the VoxCeleb creators only provide links to the
youtube videos instead of the actual video and/or audio data. This project contains a tool to re-scrape all 
of the VoxCeleb data from youtube due to this new limitation.

The first command below will iterate over the files in the dev set and download all the appropriate audio files. The
second command will do the same for the test set. 
```
python voxcelebdownloader.py --url-directory data/vox1_dev_txt --audio-directory data/audio --output-dir data/spectrograms
python voxcelebdownloader.py --url-directory data/vox1_test_txt --audio-directory data/audio --output-dir data/spectrograms
```

Downloading the dataset requires [youtube-dl](https://youtube-dl.org/) in order to download the audio data from the 
youtube video links that the VoxCeleb dataset provides. Even VoxCeleb1 can take a very long time and consists of a very 
large amount of data.

See `speakerid.py` and `voxcelebdownloader.py` to see how this was done. Unfortunately I don't know how to stop the generation
of the `*.part` files in the directory where the download command is ran from so they will have to be cleaned up
manually via `rm *.part` after the download has completed.


## Protonet

