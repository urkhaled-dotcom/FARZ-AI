# TrashNet dataset attribution

FARZ uses the five classes Glass, Paper, Cardboard, Plastic, and Metal from the TrashNet dataset created by Gary Thung and Mindy Yang. Trash is excluded. The original project requests citation when the dataset is used.

- Original project: https://github.com/garythung/trashnet
- Author dataset repository: https://huggingface.co/datasets/garythung/trashnet
- The author's Hugging Face dataset page lists a resized archive named dataset-resized.zip and identifies the dataset license as MIT.

FARZ modifications:
- selects the five agreed classes only;
- holds out six photos in three byte-identical cross-label duplicate pairs pending review;
- groups additional non-identical perceptual-hash candidates before splitting;
- creates a new stratified group split instead of using the legacy split;
- does not modify or delete source photos.

Preserve the upstream attribution and copyright/license notices when redistributing the dataset or derivatives.
