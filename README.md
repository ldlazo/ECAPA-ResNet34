# ECAPA-ResNet34
ECAPA TDNN and ResNet34 from [this paper](https://arxiv.org/pdf/2010.12468.pdf)

Implementation of ECAPA TDNN with Dynamic Dilation, except for the first directly connected group, the 7 remaining groups in the central Res2Net convolutions have dilation factors of 2, 3, 4, 5, 6, 2, and 3 respectively.

Implementation of ResNet34 that is improved with the Squeeze-Excitation blocks after each ResBlock component and the Channel dependent Attentive Statistics pooling layer.

ECAPA TDNN

![ECAPA](https://user-images.githubusercontent.com/93837797/140899235-8a07cae2-d86a-4d6b-a0d6-97cbbc9dcba9.png)

ResNet34

![Res](https://user-images.githubusercontent.com/93837797/140899272-698b5143-d1c8-403a-b7bf-72a4031c213c.png)

