# Let's run the analysis again, but this time with the federalist papers
import re, nltk, os, sys, platform
from pandas import DataFrame
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager

# Word or character tokenization?
tokenizeMethod = "char" # use "word" for word based

# Types of labels for documents in the corpus
labelTypes = ('title', 'dynasty', 'siku', 'sikusub', 'author')

# Color Value:
colorValue = 2 # Index of label to use for color

# Label Valeu
labelValue = 0 # Index of label to use for labels

# Limit the number of words to look at
commonWords = 1000 # Set to None if you want to look at all words or use custom vocab

# Set the vocabulary you are interested in
limitVocabulary = None # None will not use a vocab list
# limitVocabulary = "之 了 不 的 得 人".split(" ")

# Analysis type. PCA or HCA
analysisType = "PCA"

# How many components?
pcaComponents = 2

# Point size (set to integer)
pointSize = 8

# Show point labels:
pointLabels = False

# Plot loadings
plotLoadings = True

# Hide points:
hidePoints = True

# Set the font
if platform.system() == "Darwin":
    font = matplotlib.font_manager.FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")
elif platform.system() == "Windows":
    font = matplotlib.font_manager.FontProperties(fname="SimSun")
elif platform.system() == "Linux":
    # This assumes you have wqy zenhei installed
    font = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")

# Extra modules will be loaded if you want to parse into words.
if tokenizeMethod == "word":
    try:
        import jieba
    except ImportError:
        print("For word tokenizing, you will need to install the jieba library")
        print("You can do so by running the following command:")
        print("pip install jieba")
        sys.exit()

# Function to clean the text
def clean(text):
    removeitems = "? , . ' \" 。 《 》 ， 、 【 】 ！ ？ “ ” ： ； ＜ （ ） ( ) - 「 」 〔 〕 ＞".split(" ")
    for item in removeitems:
        text = text.replace(item, "")
    text = re.sub("\s+", "", text)
    return text

# Function to tokenize the text. I've set it up so should have spaces between each token
def tokenize(text, tm = tokenizeMethod):
    if tm == "char":
        text = " ".join(list(text))
    elif tm == "word":
        # this uses the jieba library (you will need to install it), but there are other
        # good options out there, the best probably being stanford's parsers
        # this will be VERY slow
        text = " ".join(jieba.cut(text))  
    else:
        print("Set tokenizationMethod to either char or word")
    return text

print("Loading, cleaning, and tokenizing")
# Go through each document in the corpus folder and save info to lists
texts = []
labels = []

for root, dirs, files in os.walk("corpus"):
    for i, f in enumerate(files):
        # add the labels to the label list
        labels.append(f[:-4].split("_"))

        # Open the text, clean it, and tokenize it
        with open(os.path.join(root,f),"r", encoding='utf8') as rf:
            texts.append(tokenize(clean(rf.read())))
        
        if i == len(files) - 1:
            print(f"\r{i+1} of {len(files)} processed", end='\n', flush=True)
        else:
            print(f"\r{i+1} of {len(files)} processed", end='', flush=True)
# if tokenizeMethod == "word":
#     server.stop()


print("Vectorizing")
countVectorizer = TfidfVectorizer(max_features=commonWords, use_idf=False, vocabulary=limitVocabulary,  analyzer='word', token_pattern='\S+')
countMatrix = countVectorizer.fit_transform(texts)
print("Normalizing values")
countMatrix = normalize(countMatrix)
countMatrix = countMatrix.toarray()

print("Performing PCA")
# Lets perform PCA on the countMatrix:
pca = PCA(n_components=2)
myPCA = pca.fit_transform(countMatrix)

print("Setting format")
# find all the unique values for each of the label types
uniqueLabelValues = [set() for i in range(len(labelTypes))]
for labelList in labels:
    for i, label in enumerate(labelList):
        uniqueLabelValues[i].add(label)

# create color dictionaries for each of the texts
colorDictionaries = []
for uniqueLabels in uniqueLabelValues:
    colorpalette = sns.color_palette("husl",len(uniqueLabels))
    colorDictionaries.append(dict(zip(uniqueLabels,colorpalette)))

# Now we need the Unique Labels
uniqueColorLabels = list(uniqueLabelValues[colorValue])
# Let's get a number for each class
numberForClass = [i for i in range(len(uniqueColorLabels))]

# Make a dictionary! This is new sytax for us! It just makes a dictionary where
# the keys are the unique years and the values are found in numberForClass
labelForClassNumber = dict(zip(uniqueColorLabels,numberForClass))

# Let's make a new representation for each document that is just these integers
# and it needs to be a numpy array
textClass = np.array([labelForClassNumber[lab[colorValue]] for lab in labels])


# Make a list of the colors
colors = [colorDictionaries[colorValue][lab] for lab in uniqueColorLabels]

if hidePoints:
    pointSize = 0

print("Plotting texts")
for col, classNumber, lab in zip(colors, numberForClass, uniqueColorLabels):
    plt.scatter(myPCA[textClass==classNumber,0],myPCA[textClass==classNumber,1],label=lab,c=col, s=pointSize)

# Let's label individual points so we know WHICH document they are
if pointLabels:
    print("Adding Labels")
    for lab, datapoint in zip(labels, myPCA):
        plt.annotate(str(lab[labelValue]),xy=datapoint, fontproperties=font)

# Let's graph component loadings
if plotLoadings:
    print("Rendering Loadings")
    loadings = pca.components_
    vocabulary = countVectorizer.get_feature_names()
    
    for i, word in enumerate(vocabulary):
        plt.annotate(word, xy=(loadings[0, i], loadings[1,i]), fontproperties=font)
    

# Let's add a legend! matplotlib will make this for us based on the data we 
# gave the scatter function.
plt.legend(prop=font)
plt.show()