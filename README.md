# QTDU - Quality of Tissues from Dermatological Ulcers

**WARNING:** Before we get started, please notice QTDU is **NOT** a clinical software/prototype.
Repository data is for demonstration and reproducibility purposes **ONLY**.

The repository provides QTDU implementation, as well as numerical datasets and image sets used in the validation of the approach.
QTDU is currently under analysis for publication in a peer-reviewed journal.  

*	We will release full source code after the first review stage
*	Datasets include validation data we used in the experiments

QTDU is an approach that combines deep learning models to superpixel-driven segmentation methods for assessing the quality of tissues from dermatological ulcers.
The approach consists of a three-stage pipeline for the obtaining of ulcer segmentation, tissues' labeling, and pixelwise area quantification, as follows:

<p align="center">
  <img src="https://github.com/gu-blanco/qtdu/blob/master/readme_images/architecture.png" width="60%" height="60%">
</p>

QTDU was implemented in [Python 3.0](https://www.python.org/download/releases/3.0/) using third-party libraries [Tensorflow](https://www.tensorflow.org/), [Keras](https://keras.io/) and [Pandas](https://pandas.pydata.org/). 
[Weka Framework](https://www.cs.waikato.ac.nz/ml/weka/) was also used for some experimental evaluations.

### 1. Minimum Requirements

* Python 3.6.3
* Scikit-learn 0.2
* Keras 
* Tensor-flow

We suggest the use of a GPU-based cluster for speeding up the deep learning models' training time.

### 2. QTDU Implementation

QTDU implementation is divided into two main blocks, as follows: 

* Code-block:
	* InceptionV3 implementation (Link to be released after review)
	* ResNet50 implementation (Link to be released after review)

### 3. QTDU Datasets

Datasets' repository is organized as follow:

*	Labeled superpixel images
	* 44,893 image superpixels
	* Metadata.csv, which provides labels for the image superpixels

* .arff datasets
	* ColorLayout16.arff, which provides superpixels representation regarding MPEG-7 Color Layout extractor (16-dimensional vectors),
	* ColorStructure128.arff, which provides superpixels representation regarding MPEG-7 Color Structure extractor (128-dimensional vectors), 
	* ScalableColor256.arff, which provides superpixels representation regarding MPEG-7 Scalable Color extractor (256-dimensional vectors), 
	* ColorStructureKG.arff, which provides superpixels representation regarding MPEG-7 Color Structure extractor reduced by PCA according the Keiser-Guttman criterion (31-dimensional vectors), 
	* ScalableColorKG.arff, which provides superpixels representation regarding MPEG-7 Scalable Color extractor reduced by PCA according the Keiser-Guttman criterion (18-dimensional vectors),
	* ColorStructureSP.arff, which provides superpixels representation regarding MPEG-7 Color Structure extractor reduced by PCA according the Scree-Plot criterion (12-dimensional vectors), 
	* ScalableColorSP.arff, which provides superpixels representation regarding MPEG-7 Scalable Color extractor reduced by PCA according the Scree-Plot criterion (6-dimensional vectors),

* .csv datasets
	* ColorLayout16.csv, which provides superpixels representation regarding MPEG-7 Color Layout extractor (16-dimensional vectors),
	* ColorStructure128.csv, which provides superpixels representation regarding MPEG-7 Color Structure extractor (128-dimensional vectors), 
	* ScalableColor256.csv, which provides superpixels representation regarding MPEG-7 Scalable Color extractor (256-dimensional vectors), 
	* ColorStructureKG.csv, which provides superpixels representation regarding MPEG-7 Color Structure extractor reduced by PCA according the Keiser-Guttman criterion (31-dimensional vectors), 
	* ScalableColorKG.csv, which provides superpixels representation regarding MPEG-7 Scalable Color extractor reduced by PCA according the Keiser-Guttman criterion (18-dimensional vectors),
	* ColorStructureSP.csv, which provides superpixels representation regarding MPEG-7 Color Structure extractor reduced by PCA according the Scree-Plot criterion (12-dimensional vectors), 
	* ScalableColorSP.csv, which provides superpixels representation regarding MPEG-7 Scalable Color extractor reduced by PCA according the Scree-Plot criterion (6-dimensional vectors),


### 4. Additional Information and Legal Note

*Third-party codes are subject to their own restrictions and licenses.*

*Data in this repository are provided 'as is' and any expressed or implied warranties, including but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the authors of this software or its contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including but not limited to, procurement of substitutive goods or services, loss of use, data, or profits; or business interruption) however caused and on any theory of liability wheter in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this demonstration, even if advised of the possibility of such damage.*

Again, QTDU is **NOT** a clinical software/prototype.