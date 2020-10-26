# Defending-Adversarial-Examples-via-DNN-Bottleneck-Reinforcement
Code of Defending Adversarial Examples via DNN Bottleneck Reinforcement

1.data

1.1 dataset

We use three classical image dataset to train our model.

MNIST:http://yann.lecun.com/exdb/mnist/

Cifar10:https://www.cs.toronto.edu/~kriz/cifar.html

ImageNet:http://www.image-net.org/

1.2 adversarial examples

We use FGSM SGD to generate adversarial example for adversarial training.

During testing period, we use FGSM BIM C&W DNN to test the robustness of our model.

2.attack

To confuse the CNN, many method proposed to cheat add small perturbation 

3.defense

