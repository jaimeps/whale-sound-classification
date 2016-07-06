## Whale Sound Detection

<p align="center">
	<img src="https://github.com/jaimeps/whale-sound-classification/blob/master/images/logos.png" width="450">
</p>

Team: [J. Helgren](https://github.com/jhelgren), [J. Pastor](https://github.com/jaimeps), [A. Singh](https://github.com/Abhishek19895)

### Description:
In this project we analyze [The Marinexplore and Cornell University Whale Detection Challenge](https://www.kaggle.com/c/whale-detection-challenge), where participants were tasked with developing an algorithm to correctly classify audio clips containing sounds from the North Atlantic right whale.
<p align="center">
	<img src="https://github.com/jaimeps/whale-sound-classification/blob/master/images/marinexplore_kaggle.png" width="350">
</p>

The focus of our analysis is on the [winning entry (by Nick Kridler and Scott Dobson)](https://github.com/nmkridler/moby), whose methodology combines contrast-enhanced spectrograms, template matching, and gradient boosting.

Using Python along with the R interface to h2o, we reproduce the winner’s algorithm, explain its multiple components in IPython Notebook tutorials, test the results, and fine tune the classifier.

### Data:
The Kaggle training set includes approximately 30,000 labeled audio files. The test set includes approximately 54,000 files. Each file encodes a two second monophonic audio clip in AIFF format with a 2000 Hz sampling rate. 

### Project Report:
Available upon request.

### Examples:
Among the techniques explained in the tutorials (Ipython notebooks), we can highlight:
- Contrast enhancement and noise filtering, to enhance the signal of the whale call in the spectrogram
<p align="center">
	<img src="https://github.com/jaimeps/whale-sound-classification/blob/master/images/image_processing.png">
</p>
- Template matching
<p align="center">
	<img src="https://github.com/jaimeps/whale-sound-classification/blob/master/images/template_matching.png" width = 550>
</p>

### References:
- [OpenCV Template Matching](http://docs.opencv.org/3.1.0/d4/dc6/ tutorial_py_template_matching.html)
- [Scikit-Image Docs - Module Skimage Exposure](http://scikit-image.org/ docs/dev/api/skimage.exposure.html)
- [Scipy Signal - Docs](http://docs.scipy.org/doc/scipy/reference/signal.html)
- [The Marinexplore and Cornell University Whale Detection Challenge](https://www.kaggle.com/c/whale-detection-challenge)
- [Whale Detection Challenge Code](https://github.com/nmkridler/moby)
- Mark A. McDonald and Sue E. Moore, *Calls recorded from north pacific right whales (eubalaena japonica) in the eastern bering sea, Journal of Cetacean Research and Management 4 (2002), no. 3, 261–266*
- Sandra L Harris Robert J. Schilling, *Digital signal processing using Matlab*