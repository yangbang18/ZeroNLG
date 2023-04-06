# root paths to load raw images or videos
image_video_root = {
    'coco': 'data/MSCOCO',
    'flickr30k': 'data/Flickr30k',
    'msrvtt': 'data/MSRVTT',
    'vatex': 'data/VATEX',
}

num_frames = 8 # the number of frames to be uniformly sampled for each video

annotation_root = 'data/annotations'
corpus_root = 'data/corpus'

# generation settings for visual captioning in different languages
auto_settings = {
    'en': dict(
        max_length=20,
        min_length=3,
        repetition_penalty=1.0,
    ),
    'zh': dict(
        max_length=30,
        min_length=3,
        repetition_penalty=1.0,
    ),
    'de': dict(
        max_length=15,
        min_length=3,
        repetition_penalty=2.0,
    ),
    'fr': dict(
        max_length=20,
        min_length=3,
        repetition_penalty=2.0,
    ),
}
