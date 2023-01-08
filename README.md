# AIN311-Project-EyeTracking
This is repository is for AIN311 Course Project "Eye Tracking and Prior Knowledge" 

# 1. Problem Definition

The initial problem definition for our research stemmed from an idea
which came to us from our part-time research work which involved working
with IR (Infrared) sensors to create an Eye Tracking based electric
wheelchair for paralysed patients. Working with the eye tracking
sensors, we have thought about a question which was the following: "Can
we detect a change in the eye patterns of two different people reading
the text?" which was going to depend on whether we could detect a
significant difference in eye tracking pattern of two subjects with
different levels of prior knowledge on an identical sample text. This
initial question then became a more fully-fledged research question as
the following:

*"Can we train an AI model to accurately predict our subjects’ prior
knowledge on certain topics such as history, sciences, or culture given
the topic of the sample text given during the experiment?"*

# 2. Related work

On Google Scholar, we were able to find a handful of papers which had a
similar topic to ours. One was a paper by Ho, Hsin Ning Jessie, et al.
In the paper, the researchers analyzed the eye movement patterns of
students’ during the reading of a scientific text with diagrams by
creating heat maps to identify regions of interest alongside with
scanning paths to detect inter-scanning transitions between portions of
the text or diagrams. Using such metrics they have grouped the students
into high and low prior knowledge groups. But the paper used no machine
learning methods to automate this classification task, and instead
identified key differences between two groups which was the focus of the
paper. There were also a handful of works where researchers studied the
relationship between reading pattern and some other high level metric
such as attention and comprehension.

Another similar work was a paper by Cole, Michael J., et al. Where
researchers employed a Random Forest model to predict prior knowledge
similar to our intentions, but they had used cognitive effort measures
instead. The researchers tested the relationship between self reported
comprehension scores and the output of the learning model.

# 3. Methodology

Owing to the fact that the aim of this project is to test a real-life
hypothesis, our methodology is two-folds: an experiment where we collect
our data, and a machine learning component where we analyze our results
and train a ML model to try to accurately predict prior knowledge from
reading patterns.

## Layout of the Experiment

The setup of the experiment is as the following: A laptop (with eye
tracking software package installed) to display the sample texts, a
commercially available Tobii EyeX IR eye tracker, a seat and a table.

The data will be collected from students on Hacettepe University Beytepe
campus. We plan to place our aforementioned setup in a crowded place
such as a café, before we go around and ask people if they are willing
to participate in our experiment. Potential participants will be told
that they will remain anonymous, and that the sensor does not collect
their images, but rather, their eye movements. We will also promise the
students a small treat such as a cookie if they wish to participate, in
order to act as an incentive.

If the participant is willing to participate, they will be seated in
front of our setup and given a brief overview of the experiment, and
will be specifically told that they will not be graded or tested on how
fast they can read, in order to encourage a focus on a good
comprehension of sample texts. We hope that this will also encourage a
casual reading style similar to how one would read an article on the
internet.

After the verbal introduction is done, a calibration phase is conducted,
which is provided in the eye tracker’s own software package. During the
calibration, eye tracker is calibrated for the subject by having them
follow a circle on the screen for about 30 seconds, after which a
calibration score out of 5 is displayed. If a calibration score lower
than 3 is attained, the calibration is repeated until a satisfactory
calibration is achieved.

Then, the subject will be shown three 100 word-long sample texts on
French Revolution, Moai Statues, and the World Cup. All texts are
extracted from their respective Wikipedia pages. Moai and World Cup
texts are extracted from their simple English versions whereas French
Revolution text is taken from the regular English Wikipedia article in
order to have varying degrees of vocabulary complexity with the sample
text on French Revolution including some lesser known terms such as
*liberté, égalité*, and *fraternité*.

When the subject is finished reading all three texts, they will be asked
on the prior knowledge of the topics they have just read about (French
Revolution, Moai Statues, and the World Cup) and be asked to give
themselves a score ranging from 1 to 5. At this point the experiment is
concluded, and we will thank the participant for their valuable time and
offer them the treat they were promised.

## Experimental Variables

### Independent Variable

Our dependent variable is the prior knowledge on the subject matter of
the sample text. This data will be collected from the subjects after
they are done reading the sample text. This will be done by asking the
subjects to rank their knowledge on the subject of the text they have
just read. The variable will be categorical, ranging from 1 through 5, 1
being no familiarity with the subject matter, whereas 5 corresponds to
knowing most of the information presented in the sample paragraph.

### Dependent Variable

Our dependent variable is the eye movement patterns of the subjects
collected by the IR eye-tracking sensor during the reading of sample
texts. This data is collected as the location of eye fixations on the
laptop screen in the form of coordinates (x,y). Naturally, this data
will be combined with the bounding boxes of the individual words on the
screen which will be extracted with an optical character recognition
tool (OCR) such as pyTesseract.

# 3. 2. Machine Learning

### Type of ML Model

We will be using Naive Bayes, SVM, and Random Forest machine learning
models as the core of this ML Research Project. We will use naive Bayes
because we have the most experience with it and it fits well with our
bag-of-words representation, and our hypothesis which assumes that
people with different levels of prior knowledge will fixate at different
clusters of words when reading the same text.

We also plan to use SVM because we anticipate that our sample size will
be small, around 50, and dense with high dimensionality, as our data
points will be extracted from around 300 to 600 eye fixations each
according to our preliminary tests. The SVM algorithm is known to
perform well in situations with high dimensional data.

Finally, we also chose to use Random Forest in case the feature space we
obtain from our data collection cannot be efficiently separated by a
hyperplane as SVM does, and Random Forest is also known to perform well
with high dimensional data similar to SVM.

### Data Preprocessing

We are using a UI provided by Tobii in order to collect data, and the
output from the UI is in .json format which includes unneeded data that
is crucial for the working of the sensor. Because of this, we will run
our data preprocessing steps on this raw .json file after which we are
only left with a list of eye fixations and their coordinates on the
screen in the (x,y) format. It should also be noted that there are two
types of eye motions collected by the sensor (saccade and fixation), so
we will manually filter out the saccades and other sensor-related
statues signals such as the "heartbeat" signal which checks if the
sensor is still connected to the laptop or not.

### Loss Function

As for our loss function, we currently wrote custom code in Python to
generate a bag-of-words representation using Python’s dictionary
datatype to be used by Naive Bayes algorithm. This bag-of-words
representation will also be converted to 1-D vector representations for
SVM and Random Forest algorithms. By storing the collected data in this
format, we can easily use any difference or similarity metric such as
cosine similarity (which we created a custom version of for this
experiment) as we have two 1-D vectors of, let’s say, two different
reading by two different subjects.

### Hyperparameters

One possible hyperparameter is the distance metric whilst calculating
the vector magnitude in the cosine similarity algorithm mentioned above.
The SVM model has C and gamma hyperparameters, and Random Forest has
`n_estimators`, `min_samples_split`, `min_samples_leaf`, `max_features`,
`max_depth`, and  
`bootstrap` hyperparameters. The hyperparameter tuning will be conducted
after we implement the models, and the most suitable hyperparameters
will be investigated accordingly.

# 4. Experimental evaluation

The evaluation metrics that we plan to use to evaluate the performance
of our models are the following.

### Accuracy

This is the fraction of predictions that the model got correct. It is
defined as the number of correct predictions divided by the total number
of predictions.

### F1 score

This is the harmonic mean of precision and recall. It is defined as the
harmonic mean of precision and recall.

### AUC-ROC

This stands for "Area Under the Receiver Operating Characteristic
curve." It is a measure of the model’s ability to discriminate between
positive and negative cases.

### Confusion matrix

This is a table that shows the number of true positive, true negative,
false positive, and false negative predictions made by the model.

### Classification report

We also intend to create a summarized report card involving the metrics
above separately for three sample texts (French Revolution, Moai
Statues, and the World Cup).

# 5. Bibliography

1.  Cole, Michael J., et al. "Inferring user knowledge level from eye
    movement patterns." Information Processing & Management 49.5 (2013):
    1075-1091.

2.  Ho, Hsin Ning Jessie, et al. "Prior knowledge and online
    inquiry-based science reading: Evidence from eye tracking."
    International journal of science and mathematics education 12.3
    (2014): 525-554.

3.  Kaakinen, Johanna K., Jukka Hyönä, and Janice M. Keenan. "How prior
    knowledge, WMC, and relevance of information affect eye fixations in
    expository text." Journal of Experimental Psychology: Learning,
    Memory, and Cognition 29.3 (2003): 447.
