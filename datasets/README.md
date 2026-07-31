# Dataset Preparation

This directory stores the local image and annotation files required for the benchmark. While Hugging Face datasets (like CVBench and MMStar) are downloaded automatically by the framework, datasets like MS COCO and RePOPE must be downloaded manually.

Please run the following commands from the root of this `datasets/` directory to download and extract the necessary files.

## 1. MS COCO Dataset (2017)

Download the COCO 2017 validation images and annotations:

```bash
mkdir coco
cd coco

# Download and extract images
wget [http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
unzip val2017.zip
rm val2017.zip

# Download and extract annotations
wget [http://images.cocodataset.org/annotations/annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

# Clean up unnecessary train files to save space
rm annotations/*_train2017.json
cd ..
```

## RePOPE Dataset

RePOPE utilizes the MS COCO 2014 validation images but provides custom annotations.


``` Bash
mkdir repope
cd repope

# Download and extract COCO 2014 images
wget [http://images.cocodataset.org/zips/val2014.zip](http://images.cocodataset.org/zips/val2014.zip)
unzip val2014.zip
rm val2014.zip

# Download and extract standard COCO 2014 annotations
wget [http://images.cocodataset.org/annotations/annotations_trainval2014.zip](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
unzip annotations_trainval2014.zip
rm annotations_trainval2014.zip

# Clean up unnecessary train files to save space
rm annotations/*_train2014.json

# --- Download Custom POPE and RePOPE Annotations ---

# 1. Standard POPE Annotations
mkdir pope_annotations
cd pope_annotations
wget [https://raw.githubusercontent.com/YanNeu/RePOPE/refs/heads/main/annotations/coco_pope_random.json](https://raw.githubusercontent.com/YanNeu/RePOPE/refs/heads/main/annotations/coco_pope_random.json)
wget [https://raw.githubusercontent.com/YanNeu/RePOPE/refs/heads/main/annotations/coco_pope_popular.json](https://raw.githubusercontent.com/YanNeu/RePOPE/refs/heads/main/annotations/coco_pope_popular.json)
wget [https://raw.githubusercontent.com/YanNeu/RePOPE/refs/heads/main/annotations/coco_pope_adversarial.json](https://raw.githubusercontent.com/YanNeu/RePOPE/refs/heads/main/annotations/coco_pope_adversarial.json)
cd ..

# 2. RePOPE Annotations
mkdir repope_annotations
cd repope_annotations
wget [https://raw.githubusercontent.com/YanNeu/RePOPE/refs/heads/main/annotations/coco_repope_random.json](https://raw.githubusercontent.com/YanNeu/RePOPE/refs/heads/main/annotations/coco_repope_random.json)
wget [https://raw.githubusercontent.com/YanNeu/RePOPE/refs/heads/main/annotations/coco_repope_popular.json](https://raw.githubusercontent.com/YanNeu/RePOPE/refs/heads/main/annotations/coco_repope_popular.json)
wget [https://raw.githubusercontent.com/YanNeu/RePOPE/refs/heads/main/annotations/coco_repope_adversarial.json](https://raw.githubusercontent.com/YanNeu/RePOPE/refs/heads/main/annotations/coco_repope_adversarial.json)
cd ../..
```

## GridVQA Dataset

The [GridVQA dataset](https://github.com/deephelms28/GridVQA-X/tree/main) can be downloaded from [HuggingFace](https://huggingface.co/datasets/Aikyam-Lab/gridvqa-dataset) using

```bash
cd datasets
mkdir gridvqa
cd gridvqa
hf download Aikyam-Lab/gridvqa-dataset --repo-type dataset --local-dir gridvqa/

```