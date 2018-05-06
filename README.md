# R-Net-in-CNTK
- A CNTK implementation of [R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS](https://www.microsoft.com/en-us/research/publication/mrc/). This project is designed for the [MS MARCO](http://www.msmarco.org/) dataset.
- The data processing is the same as [BiDAF's code from MSRA](https://github.com/Microsoft/CNTK/tree/nikosk/bidaf/Examples/Text/BidirectionalAttentionFlow/msmarco/team_xyz/script)
- The part of ploymath's output still has some bugs, because of the material of CNTK is finite. But you can temporary use BiDAF's output-layer.

#Requirements
- Python >= 3.5
- cntk >= 2.5.1

#Usage
This is similar to [BiDAF's code from MSRA](https://github.com/Microsoft/CNTK/tree/nikosk/bidaf/Examples/Text/BidirectionalAttentionFlow/msmarco/team_xyz/script)
