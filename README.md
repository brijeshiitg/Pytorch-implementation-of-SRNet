# Deep Residual Network for Steganalysis of Digital Images (SRNet model) Pytorch Implementation:

Model pretrained weights are in *checkpoints* directory
The model is trained on the S-Uniward 0.4bpp in the same setting as reported in the paper:
"Deep Residual Network for Steganalysis of Digital Images"
The model can be tested using the file test.py
The tensorflow code of the same can be found at: http://dde.binghamton.edu/download/feature_extractors/

The test accuracy reported in the paper is **89.77%**. My implementation achieved **89.43%** on S-Uniward 0.4bpp.

The model is trained and tested on Tesla V-100-DGX with 32GB GPU.



<table>
  <tr>
    <td align="center">SRNet architecture</td>
  </tr>
  <tr>
    <td valign="top"><img src="srnet.png"></td>
  </tr>
 </table>

 ### Update: all the files used for training has been updated
