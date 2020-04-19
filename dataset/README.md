# Pupil Finder: Dataset

This dataset contains infrared images in low and high resolution, all captured in various lightning conditions and by
 different devices. The images are not in this repository and must be [downloaded](http://mrl.cs.vsb.cz/eyedataset)
 separately.
 
```
dataset
├── s0001
|   ├── s0001_00001_0_0_0_0_0_01.png
|   ├── s0001_00002_0_0_0_0_0_01.png
|   └── ...
├── s0002
├── ...
├── annotation.txt
├── pupil.txt
├── stats_2018_01.ods
└── README.md
```

## Pupil points

`pupil.txt` contains the annotation for approximately 15000 pupil points (images) and can be downloaded from [here](http://mrl.cs.vsb.cz/data/eyedataset/pupil.txt).
The format used to store the points is the following:

```
s0014/s0014_03817_0_0_1_0_0_01.png 36 42
s0014/s0014_03818_0_0_1_0_0_01.png 42 47
s0014/s0014_03819_0_0_1_0_0_01.png 39 44
...
```

Where the first column is the path of the image, and the second and third columns are the location of the pupil.

## Acknowledgements

The dataset was obtained [here](http://mrl.cs.vsb.cz/eyedataset) from the Media Research Lab (MRL) at the
 Department of Computer Science, Faculty of Electrical Engineering and Computer Science, VSB - Technical University
 of Ostrava.