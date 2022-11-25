# research
Abstract - A technique called face recognition uses a person's face to recognize or confirm their identification. An image's subject can be recognized using face recognition techniques on screen, and in real-time. People that hide their faces are engaging in criminal activity. People can commit crimes and are able to hide from the camera, because of the covered face and it becomes difficult to identify the person. We are going to propose a methodology to reconstruct faces from occluded masked images. We systematically investigate model scaling, which can improve performance by carefully balancing network depth, breadth, and resolution. Based on this fact, we propose a unique scaling method that scales all depth, breadth, and resolution parameters equally using a simple yet very effective compound coefficient. The proposed model extract features using the EfficientNet algorithm and then features are used for the segmentation. The segmentation is done using the EfficientDet algorithm and then we apply the ADAM classifier for the classification purpose. In our proposed model, we use ReLU as an activation function and we compare our results with publically available models. Then lastly, we will get a complete face image without a mask as an output of the proposed model.

[1]	Introduction

A method for recognizing or Facial recognition is the process of authenticating someone's identity by looking at their face. Facial recognition technology can identify people in both images and video. Using a facial recognition system, a digitized picture or video frame of a human face may be compared to a database of human faces. Typically, face recognition doesn't need a huge database of photographs to identify a person. These technologies extract certain, recognizable features from a person's face using computer algorithms. Then, these characteristics like the distance between the eyes or the shape of the chin are represented mathematically.

Due to the increased crime rate, there are cameras almost everywhere right now for tracking. The problem is that criminals are increasingly hiding their faces when committing crimes in order to avoid being caught on surveillance cameras. They may pass by unnoticed because mask usage is becoming more widespread, thus no one would suspect them of committing any crimes [2]. Because of this, finding the perpetrator is more difficult. Therefore, to correctly identify the person from occluded masked photos, a face reconstruction approach is needed.

A machine learning method called deep learning instructs computers to carry out activities that people do automatically. Deep learning is a crucial component of driverless automobiles' ability to recognize a stop sign or separate a pedestrian from illumination. By identifying complex patterns in the data they analyze, deep learning networks acquire new knowledge. The networks may create computational models with several processing layers to explain the data at various levels of abstraction.
The large-pose variation, a common face recognition obstacle, is the most crucial component. The inability of manually crafted feature descriptors to successfully extract the identity discriminative feature limits the effectiveness of face recognition systems. Deep learning-based computers have already greatly outperformed humans in facial recognition on a number of different parameters. [1].

From digital images and videos, computers and algorithms may extract important information, and other visual inputs using a branch of artificial intelligence known as computer vision, and then utilize that knowledge to act or make suggestions. Images can be segmented, objects can be identified, faces can be recognized, edges can be detected, patterns can be found, images can be classified, and features may be matched, among other computer vision techniques.

A model must determine key points that correspond to areas or landmarks on a human face, such as the eyes, nose, lips, and other features, for the job of "facial landmark recognition" in computer vision. Face landmarks are fundamental features of a human face that give us the ability to recognize various faces. In the suggested study, 128 facial landmarks are utilized.

To recognize and understand an image's pixel-level information, the narrator uses image segmentation. Unlike to object identification, where item bounding boxes may overlap, every pixel in an image belongs to a single class. To discriminate between zero- and non-zero values in binary, employ a mask. When a mask is applied to another binary or grayscale picture of the same size, all pixels that are zero in the mask are likewise set to zero in the output image. The remaining items don't change.

A broad range of issues with image production includes inpainting. Filling up the blank pixels is the aim of inpainting. Deblurring, denoising, and artifact removal are some of the operations that are included in what may be considered the creation or modification of pixels. In order to portray a full image, damaged, deteriorating, or missing portions of the artwork are filled in throughout the conservation process. This method is frequently employed in picture restoration. 

An algorithm for gender classification predicts a person's gender using an image of their face. A solid gender classification approach can help many other systems, such as face recognition and intelligent human-computer interactions, perform better. This method allows for the exceptionally effective collection of important data.

A compound coefficient is used for each depth, breadth, and resolution parameter evenly scaled using the EfficientNet convolutional neural network design and scaling technique. To quickly and easily scale up models, EfficientNet uses a method known as a compound coefficient. Compound scaling uses a currently defined set of scaling factors to consistently scale that dimension rather than randomly increasing width, depth, or resolution. The EfficientNet models perform better than conventional CNNs in terms of accuracy and efficiency by a factor of orders of magnitude by lowering parameter size and FLOPS.

For deep learning model training, Adaptive Moment Estimation (Adam) is an additional optimization approach to stochastic gradient descent after analyzing the first and second moments of the gradient, the learning rate of every weight in the neural network is modified. Adam has the benefit of being able to work with sparse gradients and naturally do a certain type of melting. Its parameter updates' magnitudes are unaffected by gradient rescaling, and the stepsize hyperparameter roughly determines how big the steps are.

The Rectified Linear Unit is the deep learning activation function that is unutilized most frequently (ReLU). The function returns 0 if the input is negative; however, The value is returned if the input is positive. The plot of the function and its derivative is used to define the function. ReLU and its derivative storyline. The ReLU equation is f(x) = max (0, x) ReLU is the default activation function and the most often used activation function in neural networks, particularly CNN. The ReLU formula is:
f(x) = max (0, x)


We're going to suggest a model that can rebuild faces from obscured images. The entire assignment is broken down into four primary components, therefore the full model is built on four sub-modules, including a module for identifying face landmarks, a module for segmenting masks, a module for inpainting, and a module for identifying gender. This model will utilize a masked picture as its input, extract features from the image using the Efficient Net method using a 128 landmark point technique, segment the image using the retrieved features using the EfficientDet algorithm, and finally perform classification using the Adam classifier. The end output of the proposed model will be a whole face image devoid of a mask.

[1]	Literature Review
Yu et al [3] offered generative image in-painting using a using a particular contextual attentiveness (CA) layer, a two-stage GAN-based network that transfers identical feature patches from surrounding major visible areas to the unoccupied regions. The copy-paste technique has the potential to introduce unwanted artifacts into the recovered parts, despite the possibility of network-wide end-to-end training.

Song et al. [4] introduced the two-stage network used to finish a face called the Geometry Aware Face Completion (GAFC) model. The first stage is determining the facial geometry of the face using a facial geometry estimator. Using the facial geometry data from the second step, The face is completed with an encoder-decoder generator. Considering that the model is already skilled in network extraction. It was able to outperform a number of existing face completion algorithms, although at a large computational expense.

Nazeri et al. [5] suggested an After the unwanted items have been eliminated, the original image can be restored using a GAN-based Edge-Connect approach (EC). The issue is divided into two phases by EC: image completion and an edge generator Using the facial geometry data from the second step, an encoder-decoder generator completes the face. Using hallucinated edges following edge generators of various sorts and the edges of the missing section, the gaps are filled by the image completion network. Due to EC, better results were obtained and the missing sections could be restored. It cannot provide a precise edge map when large pieces are missing.

PCA reconstruction and iterative error correction methods can be used to remove eyeglasses from facial images [6] but this is a non-learning algorithm, and are limited to removing menial things from images and are not able to learn. Satoshi Iizuka [7] illustrates how to fix defective parts in an image and eliminate irrelevant elements from it by using a GAN setup with two classifiers. Learning-based algorithms for image editing are useful for erasing items with little structural and visual variations. However, because of the item's enormous size and complexity, these approaches do not work well for painting veiled faces.
    
Din et al. [8] presented work to rebuild the area covered by the medical face mask after the mask has been removed, a two-stage framework based on GAN is used (MRGAN). After the masks have been found in the first stage, the faces are gained in the second. The experiment results performed better than earlier image-editing methods. On the other hand, this method requires a lot of time and effort. Additionally, this approach does not work well for a variety of things (occluded face objects).

To create true frontal view face images, many GAN variant-based synthesis approaches have been presented [9]. The encoder retrieves identity-preserved elements from the target posture, which the posture code [10] governs, and landmark heatmaps [11], and appearance flow [12] in the convolutional neural network architecture-based encoder-decoder that is widely employed with GAN generators.

Khan et al. [13] developed a GAN-based two-stage network for removing microphones. When small details are eliminated, the results are plausible, but when vast, intricate missing pieces are removed, the results are fabricated.

[2]	PROPOSED SOLUTION:
2.1)	METHODOLOGY:

Our proposed methodology is split up into different modules. Each module performs a specific task to get the required output. There is a module that can predict the landmarks of the face, a module to perform mask segmentation, an inpainting module, and a module for gender classification.

The module for landmark prediction will anticipate 128 landmark sites using a masked face image are more interested in the underlying topological structure, position, and expression of the landmarks when reconstructing recognized mask sections of an image than in their precise locations. This ground-breaking prediction model is based on the layered HG architecture proposed by Bulat et al. Each HG's heatmap is checked against the heatmap of the actual environment. When used for training, adaptive wing loss may change its form to accommodate various kinds of ground truth heatmap pixels. This flexibility penalizes foreground loss more harshly than background loss. The disparity between the pixels in the front and background was addressed using The front is given a lot of weight in the Weighted Loss Map, while the background is given a lot of difficulties.

In the binary segmentation map that the mask segmentation module generates, the mask object is represented by 1 and the remaining pixels in the picture are represented by 0. The network design for the mask segmentation module employed in this study is based on the EfficientDet technique, which is used to categorize and detect unique elements in an image. A class name and bounding box coordinates are assigned to identifiable objects in an image during object detection. In addition, each item has an object mask created by the EfficientDet algorithm based on the traits gathered by the EfficientNet method.

By using two models—the discriminator and the generator—with inputs of masked face images and their landmarks, the inpainting module attempts to complete faces. The network consists of a long block for short-term attention, seven blocks for residuals with dilated convolutions, and three blocks for progressively down-sampled encoding. Then, expanding the receptive field using stacked dilated blocks and connecting temporal feature maps with the long-short attention layer, the feature maps are gradually up-sampled to the decoder's input resolution to incorporate more data. The relevant encoder and decoder layers are also connected through shortcuts. The channel also carries out To changes in the weights of the shortcut and final layer features, 11 convolution operations are performed before each decoding layer. As a result, both in terms of their geographic and temporal size, the network could employ remote attributes more frequently. The generator's structure is shown via the inpainting module.
 


Faces tend to vary even slightly from person to person in order to avoid being recognized by the enemy. The generator depends on the notion that facial reconstructions are created by finding landmarks. After that, the discriminator decides if the facial reconstructions and the actual image data look comparable. Convergence happens when the created results are identical to the real ones. By including spectral normalization (SN) in the discriminator's training block, we increased the stability of the training. Additionally, a layer for managing the different features is introduced. A single provider, which gets a picture and its landmarks as input, serves as our discriminator. Three major justifications may be offered for this: The deep learning architecture's attention layer promotes attribute consistency. The dependence of the results on landmarks, which already give global structure, and the fact that the inpainting module was trained independently for men and women, which also provides consistency for gender traits, all contribute to a well-developed system.

In this work, we created a gender classification model using the Efficient Net algorithm.   FFHQ dataset was used to train this model. In order to extract features from images, we applied the EfficientNet technique. This model has been fine-tuned, and specific fully linked layers with custom node IDs have been added. We applied the Relu activation function in each densely linked hidden layer. We employed batch normalization after each layer.

2.2)	MODEL SCALING:

We are going to scale our model on the basis of depth, width, and resolution. We can use these coefficients separately but the problem is these factors rely on each other and can vary under different restraints so we are going for the compound model scaling to overcome this respected issue.
 

FIGURE:1 SCALING OF MODEL
Many convolutional networks scale the network depth as a typical method of operation. Additionally, it generalizes effectively to different tasks and may collect richer and more complicated characteristics. The vanishing gradient issue, however, makes it more challenging to train deeper networks. Despite the training problem being solved by a number of methods, including skip connections and batch normalization, the accuracy gain of a very deep network decreases.

For small-sized models, scalability of the network width is frequently employed. Wider networks are often more easily trainable and have a tendency to capture more fine-grained data but extremely wide but shallow networks frequently struggle to capture higher-level characteristics. As networks get considerably bigger, our empirical findings demonstrate this.
Input images with higher quality may be able to capture more intricate patterns. Accuracy is increased with higher resolutions, but the accuracy gain decreases at very high resolutions. It gives accurate results for 224x224, 299x299, and 331x331 dimensions.

 
FIGURE:2 WIDTH, DEPTH, RESOLUTION


2.3)	LOSS FUNCTION:

CONTRASTIVE LOSS:
We use the contrastive loss function to quantify the gap between two images. The distance between two identical image features, the distance between two different image features, and the loss function are all clearly addressed is defined as follows.

 

 Here, 
    m is the number of anchor-positive.
    Dᵢⱼ = ||f(xᵢ) — f(xⱼ)||² is the distance between deep  
features, where f(xᵢ) and f(xⱼ) correspond to the images xᵢ and xⱼ respectively.
yᵢⱼ= +/-1 is the indicator fa or similar label,
 [.]⁺ is the hinge loss function max (0,)

MEAN SQUARED ERROR (MSE):
The MSE loss function is used to evaluate the pixel-level difference between images. The pixels of the predicted image is compared to the pixels of the target image to determine how far apart they are. It squares the mean of each pixel's difference.

 
Here, 
yi is the ith observed value.
ŷi is the corresponding predicted value.
n = the number of observations.


PERCEPTUAL LOSS:
The same image may be compared across multiple resolutions, or it may be compared between two distinct images that share a similar appearance. The perceptual loss function contrasts significant perceptual and semantic changes between pictures, whereas pixel-wise loss methods provide a huge error value.

 

Here,
V_j(Y) represents the activations of the jth layer.
(C_j, H_j, W_j) represents the shape of the jth layer.




[3]	DATASET
3.1)	Dataset Description:
There is a publicly accessible dataset of human faces called Flickr-Faces-HQ (FFHQ). 70 thousand png-format images from NVIDIA under the Open Source BY NC-SA 4.0 license are included in FFHQ. Each image has a size of 1024 × 1024 and features a variety of people of different ages, races, settings, and dresses, including hats and eyewear.

 
FIGURE:5 Sample from the FFHQ

3.2)	Pre-processing:
The pre-processing step includes data augmentation to increase the amount of data. It involves the process of rotation and shearing only because the dataset needs no other pre-processing rather than that.

3.3)	EDA:
We have designated the first 60,000 images to be used for training and the last 10,000 images to be used for validation in use cases that call for distinct training and validation sets. Nevertheless, we trained on all 70,000 photos. We have specifically ensured that the dataset itself contains no duplicate photographs. However, if we extracted numerous distinct faces from a single image.

 

FIGURE: EVALUATION OF DATASET

[4]		EXPERIMENTS

5.1)		EXPERIMENTAL TECHNIQUES:

5.1.1)	EFFICIENTNET ALGORITHM:

 
In general, for the model to achieve both higher accuracy and better efficiency in order to better demonstrate the effectiveness of our scaling method we are using the EfficientNet algorithm. the total number of layers in EfficientNet-B0 the total is 237 and in EfficientNet-B7 the total comes out to 813. the total number of layers in EfficientNet-B0 is 237 and in EfficientNet-B7 the total comes out to 813. It has 11M trainable parameters. Through AutoML and model scaling, accuracy and efficiency are improved. When new resources are made available, convolutional neural networks (CNNs) are frequently scaled up to obtain greater accuracy after being constructed at a fixed resource cost.
      
                           
FIGURE: WORKING OF EFFICIENTNET ARCHITECTURE

5.1.2)	EFFICIENTDET ALGORITHM:

EfficientDet, is an object detection model that takes advantage of optimizations and backbone adjustments, which also combines a BiFPN and a compound scaling approach to scale all backbones, feature networks, and box/class prediction networks simultaneously in terms of resolution, depth, and width. EfficientDet follows a one-stage-detection paradigm. A pre-trained EfficientNet backbone is used with BiFPN as the feature extractor. BiFPNN takes {P3, P4, P5, P6, P7} bidirectional feature fusion is regularly used to combine features from the EfficientNet backbone network.

 

FIGURE: WORKING OF EFFICIENTDET ARCHITECTURE

5.1.3)	ADAPTIVE MOMENT ESTIMATION (ADAM)

Adaptive moment estimation is a method for using gradient descent optimization methods. The approach is quite successful when dealing with a large problem including many data or parameters. The advantages of Adam include the fact that its stepsizes are roughly constrained by The magnitudes of parameter updates gradient rescaling invariant, and the stepsize hyperparameter does not need a stationary goal. It operates on sparse gradients and conducts step-size annealing by default.

 

FIGURE: ADAM OPTIMIZER

5.1.4) RECTIFIED LINEAR ACTIVATION    FUNCTION
The use of ReLU aids in limiting the exponential rise of computing needed to run the neural network. The computational expense of adding more ReLUs rises linearly as the CNN gets bigger.
 
FIGURE: RELU ACTIVATION

The Efficient model was used to train our FFHQ dataset. Transfer learning was used in it. 10,000 iterations were utilized to train the model at a learning rate of 2.5 x 10-4 with a batch size of two. In the end, we suffered a loss of 0.0878Our extensive testing of the inpainting model led us to the conclusion that the gender characteristic was not being preserved appropriately. The FFHQ dataset was therefore divided into subgroups for men and women. Then, using the Adam optimizer [20], we trained separate datasets for men and women. The learning rate of the discriminator is 10 x 10-5. The model was trained using four batches per subgroup for 500000 iterations.
The completely linked connected layer was trained 9,720,321 trainable parameters are available in total. With a learning rate of 256 batches of 1e-4, the Adam optimizer is utilized in the experiment. Additionally, there are a maximum of 150 epochs. With respect to the test data, our accuracy was 88.12 percent.


 
FIGURE:8 RESULT OF THE MODEL





[5]	Conclusion:
The framework for face reconstruction was developed as a result of this investigation. In order to reconstruct the face after removing the mask, a number of challenges must be solved, including mask edges on faces after removal, various mask forms, and altering angles of facial images. To obtain credible findings for complete and realistic image reconstruction, we applied image inpainting. We trained separate models for men and women for better accuracy. This solved problem with image inpainting where past attempts failed to distinguish crucial facial traits between men and women. The landmarks must be sufficiently clean and adequate, which is our basic principle., and strong to act as a guide when it comes to giving the face inpainting module structural information. To ensure attribute consistency, we created a method for gathering distant geographical feature maps and connecting them over time. The addition of landmarks makes our network more flexible, allowing us to recreate the picture with perfect form no matter what shapes we get. Our model has delivered good perceptual-quality results in the situation of major missing gaps in face photos when compared to the outcomes of earlier testing and the cutting-edge image editing methods now in use.

REFERENCES
[1]	F. Schroff, D. Kalenichenko, and J. Philbin, “FaceNet: A unified embedding for face recognition and clustering,” in Proc. CVPR, 2014, pp. 815–823.
[2]	Kevin Rawlinson. Rise in suspects using face coverings to mask identity. shorturl.at/aqAP6/, 2021. The Guardian Blog, Accessed April 16, 2021
[3]	Yu, J.; Lin, Z.; Yang, J.; Shen, X.; Lu, X.; Huang, T.S. Generative image inpainting with contextual attention. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 18–22 June 2018.
[4]	Song, L.; Cao, J.; Song, L.; Hu, Y.; He, R. Geometry-aware face completion and editing. In Proceedings of the 33rd AAAI Conference on Artificial Intelligence, Honolulu, HI, USA, 27 January–1 February 2019; pp. 2506–2513.
[5]	Nazeri, K.; Ng, E.; Joseph, T.; Qureshi, F.; Ebrahimi, M. EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning. arXiv 2019, arXiv:1901.00212.
[6]	Jeong-Seon Park, You Hwa Oh, Sang Chul Ahn, and Seong-Whan Lee. Glasses removal from facial image using recursive error compensation. IEEE transactions on pattern analysis and machine intelligence, 27(5):805–811, 2005.
[7]	Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa. Globally and locally consistent image completion. ACM Transactions on Graphics (ToG), 36(4):1–14, 2017.
[8]	Din, N.U.; Javed, K.; Bae, S.; Yi, J. A novel GAN-based network for the unmasking of masked face. IEEE Access 2020, 8, 44276–44287. [CrossRef]
[9]	T. Hassner, S. Harel, E. Paz, and R. Enbar, “Effective face frontalization in unconstrained images,” in Proc. CVPR, 2015, pp. 4295–4304 
[10]	L. Tran, X. Yin, and X. Liu, “Disentangled representation learning GAN for pose-invariant face recognition,” in Proc. CVPR, 2017, pp. 1415–1424.
[11]	Y. Hu, X. Wu, B. Yu, R. He, and Z. Sun, “Pose-guided photorealistic face rotation,” in Proc. CVPR, 2018, pp. 8398–8406.
[12]	Z. Zhang, X. Chen, B. Wang, G. Hu, W. Zuo, and E. R. Hancock, “Face frontalization using an appearance-flow-based convolutional neural network,” IEEE Trans. Image Process., vol. 28, no. 5, pp. 2187–2199, May 2019.
[13]	Khan, K.; Din, N.U.; Bae, S.; Yi, J. Interactive removal of microphone object in facial images. Electronics 2019, 8, 1115. [CrossRef]


