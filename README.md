<h1 align="center">Hi 👋, I'm Tai</h1>
<h3 align="center"> Welcome to my Breast Cancer Detection project using Machine Learning Autonomous solutions.</h3>
<br/>
🌱 This is not only a personal project, but a personal cause deep in my heart. At the time of working on this project, we lost my dear Aunt a couple of days after Christmas on the 27/12/2023. She was a true treasure who recognised the gifts in people and always spoke life and positivity into to those gifts. A true inspiration with sprinkles of wisdom. She always told me to work hard and to remmeber 'I have a seat at that table'. She was my motivation to complete this project, in continuation of her persuit to raise awareness of breast cancer and urge the importance of health and looking after our bodies, after she triumphed 8 years with the diagnosis. So join me in checking out how I used different algorithms to find an automous solution in medical imaging for breast cancer detection. This one is for my Aunt Angela who left us gifts of faith, love, joy, strength, living intentionally, and so much more.🥂 

# Breast Cancer Detection

### **Introduction** <br/>
This report delves into the details of breast cancer detection, employing various models to classify masses as either benign or malignant. The terms "benign" and "malignant" hold crucial significance in confirming a breast cancer diagnosis, with benign suggesting a non-cancerous mass and malignant suggesting a cancerous growth. The urgency of breast cancer detection goes beyond industrial applications, representing a critically urgent in the healthcare sector where manual detection processes can be time-consuming (Nallamala et al., 2019).
Globally acknowledged by the World Health Organization (WHO) as one of the leading causes of cancer-related deaths among females (Bhise et al., 2021), accounting for 25% mortality rates, cancer emphasises the heavy need for automated solutions in the healthcare industry. The question arises: How can automation help? Supervised machine learning develops as a key solution in breast cancer detection, leveraging labelled datasets encompassing characteristics of breast cancer masses to train models skilled at detecting patterns associated with benign and malignant biopsies. In the field of healthcare, these labels are typically received from biopsy results, mammograms, or other pertinent medical examinations.
The process involves extracting relevant features such as; tumour size, shape, texture, and other defining characteristics from medical images or clinical data. These features then act as input variables for a machine learning model. Some supervised learning algorithms, including logistic regression, support vector machines, decision trees, and ensemble methods such as random forests, are then deployed and trained on the labelled dataset. Throughout this training phase, the model gains the ability to map input features to their corresponding labels. Once trained, the model shows the capability to predict whether a new, unlabelled sample is benign or malignant based on its unique characteristics (features). This predictive ability significantly aids in the diagnosis of breast cancer, offering valuable insights to healthcare professionals. Automated systems supported by machine learning prove to be helpful in swiftly analysing extensive volumes of medical data, streamlining the diagnostic process, and enabling healthcare providers to concentrate on more involved tasks (Hussain et al., 2018). Supervised learning models can be further customized to individual patient data, facilitating the creation of personalised treatment plans that consider unique patient characteristics (Bhise et al., 2021).
Moreover, these machine learning models contribute largely to early detection by identifying subtle patterns indicative of malignant biopsies that are often challenging for human observers to detect. Early detection, in turn, assumes utmost importance in enhancing treatment outcomes and reducing mortality rates (Ak, 2020). The adaptability of machine learning models to continuous updates and improvements, fuelled by newly available labelled data, ensures their sustained relevance in the dynamic area of medical developments.
In summary, the deployment of supervised machine learning in breast cancer detection not only elevates diagnostic accuracy but also facilitates early detection and supports the construction of personalised treatment strategies. Therefore, this report aims to design and implement an autonomous system employing appropriate methods, tools, and models for the enhanced detection of breast cancer.

### **Packages** <br/>
Packages used throughout this report were as follows:
•	Ucimlrepo (importation of breast cancer dataset) <br/>
•	NumPy (for dataset arrays) <br/>
•	Pandas (for working with data frames) <br/>
•	Matplotlib (for plotting) <br/>
•	SciKit Learn (train/test split of dataset for model selection and evaluation metrics and scaling and encoding) <br/>
•	Plotly (for visualisations) <br/>
•	SciPy (for statistic handling) <br/>

### **Models and Techniques** <br/>
In this report, several models and statistical techniques were explored for the dataset. The models included are as follows: <br/>

•	Logistic Regression <br/>
•	Random Forest Tree <br/>
•	K-Nearest Neighbour <br/>
•	Support Vector Machine <br/>
•	Genetic Algorithm <br/>
•	Gaussian Naïve-Bayes <br/>
•	Decision Tree <br/>
•	Linear Discriminant Analysis <br/>
•	Principle Component Analysis <br/>


**The Dataset** <br/>
https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
The Wisconsin Breast Cancer dataset is comprised of features computed from digitised images of a fine-needle aspirate(FNA) of breast cancer masses. The features describe the characteristics of the cell nuclei present in the digitised image. Feature characteristics of the breast cancer masses in dataset include; radius (mean of distances from centre to points on the perimeter), ) texture (standard deviation of gray-scale values), perimeter, area, smoothness (local variation in radius lengths), compactness (perimeter^2 / area - 1.0), concavity (severity of concave portions of the contour), concave points (number of concave portions of the contour), Symmetry and fractal dimension ("coastline approximation" - 1). The dataset is available as a python import using the ‘ucimlrepo’ package in the form of metadata. For the purpose of this report the dataset was imported using this method and then converted to a data frame to enable a usable format for different python application.

Upon conducting Exploratory Data Analysis (EDA), it was found that the dataset had 569 cases in total in which 357 were benign, and 212 were malignant. This was indicated by the target column of ‘Diagnosis’. The cases that were benign accounted for 62.7% of the dataset and the cases that were malignant accounts for 37.3% of the dataset. There were 30 features in total, excluding the ID column, that determined whether a diagnosis was either benign or malignant. These characteristics are what were previously described as the characteristics of the cell mass.

**Methods** <br/>

*Dataset:* <br/>
Given that the data imported from python in a default form of metadata, the data was first converted into a data frame containing the variable information . This was done by converting the data list into a data array using the NumPy library with the target and feature names, then by converting the arrays into a data frame using the Pandas library. During the data modelling process, the target features were numerised using a dictionary and a map function. Benign biopsies were given a value of 0 from original value ‘B’, and malignant biopsies were given a value of 1 from original value ‘M’. The new numerised feature replaced the existing ‘Diagnosis’ column and this way, the column was able to be categorised and encoded to numerical formatting for model processing and for visualisation purposes. Before being passed to the model, the target feature ‘Diagnosis, was dropped from the from the data frame. This is because the target variable is what is to be predicted, so the dependent variable (target) and the independent variable (features) need to be separated. This is not always needed but given that this is a supervised machine learning task, by splitting the target and features in the dataset, the data is organised in a way that makes it easier to pass the data to the machine learning models and allows for later use of making predictions on new, unseen data. Thus, the dataset was split into variables X (features) and Y (target), with Diagnosis being the target and rest of the dataset labels being the features. Once separate, the train-test split was then defined, 80% of the data was used for training and 20% of the data was used for testing using the ‘train_test_split’ module from the Scikit Learn library. Finally, after splitting the dataset into training and testing sets, the training data was scaled using the StandardScaler() from the Scikit Learn library before being passed to the models. Standard Scaler is a preprocessing technique used to standardize the features of a dataset so that the dataset has properties of a standard normal distribution, with a mean of 0 and a standard deviation of 1. In this report, standardising the features was essential as some of the model algorithms and statistical techniques used are sensitive to the scale of input features. These models later used include, but are not limited to: K-Nearest Neighbour, Support Vector Machine, K-Means, Principle Component Analysis and Linear Discrimination Analysis. <br/>


***Standard Scaler Equation:*** <br/>

### **z=σ(x−μ)** <br/>

where z is the standardized value, x is the original value, μ is the mean of the feature and σ is the standard deviation of the feature. <br/>


**Models and  Statistical Techniques:** <br/>

*Logistic Regression* <br/>
The logistic regression model was constructed using the LogisticRegression() module from the Scikit Learn library. It was assisted with A confusion metrices for classification reporting and a learning curve graph to show how it performed with the train and test data. 

*Random Forest* <br/>
The Random Tree Forest classifier was constructed using the RandomForest() module from the Scikit Learn library up to a tree-depth of 20. When run, a randomized search for hyperparameter tuning was conducted using and estimator range of 50 to 500 and a maximum tree depth of 20. After the search was complete, the best estimator (Random Forest with the best hyperparameters) was extracted along with feature importance.

*K-Nearest Neighbour (KNN) & Gaussian Naïve Bayes (GB)* <br/>
KNN was appended with GB using the Scikit Learn library modules, KNeighborsClassifier() and GaussianNB() . For each model, the mean accuracy was measured by the average across ten folds of cross-validation and the standard deviation to give an indication of the variability of the consistency of model performance. 

*Support Vector Machine (SVM)* <br/>
The Support Vector Machine model was developed using the Support Vector Machine Classifier from the Scikit Learn library. The SVC() instance was used that includes an initializer with default hyperparameters.  The most predictive features were extracted using the SVM linear kernel.

*Genetic Algorithm* <br/>
The genetic algorithm was used to optimise feature selection for a logistic regression predictive model. The algorithm went through multiple generations, evolving its feature selection strategy to maximise accuracy. The progression of the algorithm was tracked across generations, with each generation returning a "best score" and the corresponding chromosome, representing the selected features.

*Decision Tree & Linear Discrimination Analysis* <br/>
Two classifiers, Linear Discriminant Analysis (LDA) and Decision Tree Classifier (DTC), were evaluated with hyperparameter tuning. These algorithms were composed using Scikit Learn library modules, LinearDiscriminantClassifier() and DecisionTreeClassifier().

*Principle Component Analysis* <br/>
Principle Component Analysis (PCA) was used in this report to reduce the dimensions of the data by means of feature reduction. By this, it allows the data variance to be explained to it greatest extent. Using Scikit Learn library’s PCA module, the variance was explored using a PCA component number. The variance was explained, using visual aid of a graph against the increasing number of components. PCA was then run with the Logistic Regression algorithm to predict whether dimension reduction improved model accuracy.

### **Results/Evaluation:** <br/>

*Logistic Regression* <br/>
The logistic regression model achieved an remarkable accuracy of 97.37%. The confusion matrix indicated that out of the 71 instances belonging to class 0, only 1 was misclassified, and out of the 43 instances belonging to class 1, 2 were misclassified. This resulted in a high precision of 97% for class 0 and 98% for class 1. 
The classification report provided a detailed overview, including precision, recall, and F1-score for each class. Class 0 exhibited a precision of 97%, a recall of 99%, and an F1-score of 98%. For class 1, the precision was 98%, recall was 95%, and the F1-score was 96%. The overall weighted average for precision, recall, and F1-score was 97%, indicating a robust model performance.
A learning curve suggested that as the training set size increased, the accuracy also increased. However, the train score decreased until the two scores almost meet. This is expected and shows that when the dataset is small, the model fits a small number of samples well. But as the dataset starts to increase, the model starts to generalise better and therefore increases the accuracy in the test dataset.
In summary, the logistic regression model demonstrates excellent predictive capabilities, with high precision and recall values for both classes, leading to an overall accuracy of 97.37%.

*Random Forest* <br/>
The Forest algorithm showed strong performance with an overall accuracy of 96.49%. The confusion matrix revealed a high number of true positives (40) and true negatives (70), along with minimal false positives (1) and false negatives (3). The classification report further explained the model's ability.
The precision, representing the accuracy of positive predictions, was notably high for both classes, reaching 96% for class 0 and 98% for class 1. Recall, indicated the model's ability to capture positive instances, demonstrating a remarkable 99% for class 0 and 93% for class 1.
The F1-score, combining precision and recall, underscored the model's balanced performance, giving a score of 97% for class 0 and 95% for class 1. These metrics together, signify the model's effectiveness in classifying between benign and malignant breast cancer cases, highlighting its potential as a reliable diagnostic tool.
From the randomised search performed for the best tree-depth, the Random Forest model that performed best on your dataset during the randomized search had a maximum tree depth of 15 and consisted of 271 trees. These hyperparameters are considered optimal within the specified search space, aiming to provide the best balance between model complexity and performance on the training data.

*K-Nearest Neighbour and Gaussian Naïve Bayes* <br/>
K-Nearest Neighbour (KNN) had a high mean accuracy of 96% whereas Gaussian Naive Bayes (GB) had a lower mean accuracy compared to KNN of 92%. Additionally, the standard deviations provided insights into the consistency of the models. Lower standard deviations indicated less variability in performance across different folds. The standard deviation for KNN was 0.02 and the standard deviation for the GB was 0.03

*Support Vector Machine (SVM)* <br/>
The Support Vector Machine (SVM) model achieved an accuracy of 98.24% on the test data. The precision, recall, and F1-score metrics were also evaluated for each class (0 and 1) as follows:
For class 0 (negative class):
Precision: 97%
Recall: 100%
F1-score: 99%
For class 1 (positive class):
Precision: 100%
Recall: 95%
F1-score: 98%

The overall performance metrics for the model are as follows:
Accuracy: 98.24%
Macro Average Precision: 99%
Macro Average Recall: 98%
Macro Average F1-score: 98%

The weighted average precision, recall, and F1-score take into account the class imbalance in the dataset and are also reported:
Weighted Average Precision: 98%
Weighted Average Recall: 98%
Weighted Average F1-score: 98%

The confusion matrix showed that out of 71 instances of class 0, all were correctly classified, and out of 43 instances of class 1, 41 were correctly classified. There were 2 instances of class 1 that were incorrectly classified as class 0.
In summary, the SVM model demonstrated high accuracy and good performance in correctly classifying instances for both classes, with a slightly higher emphasis on precision for class 1. The model's overall ability to discriminate between malignant and benign cases is reflected in the strong performance across various evaluation metrics. 

*Genetic Algorithm* <br/>
In the final generation, the genetic algorithm achieved a perfect accuracy score of 1.0. The associated chromosome revealed the specific features chosen by the algorithm for the optimal model. The selected features, represented in binary form, contributed to the high predictive performance of the model.
This outcome emphasises the effectiveness of the genetic algorithm in identifying a subset of features that significantly contribute to the model's accuracy. This reflected the successful convergence of the genetic algorithm towards an optimal feature set, resulting in a highly accurate predictive model for the given dataset.

*Decision Tree & Linear Discrimination Analysis* <br/>
The best hyperparameters for each classifier are reported below:

1. Linear Discriminant Analysis (LDA):
Best parameters: {'classifier__n_components': None, 'classifier__shrinkage': None, 'classifier__solver': 'svd'}
Test accuracy with the best model: 95.61%

2. Decision Tree Classifier (DTC):
Best parameters: {'classifier__criterion': 'entropy', 'classifier__max_depth': 10}
Test accuracy with the best model: 95.61%
The reported test accuracies demonstrate the performance of each classifier with the optimized hyperparameters. Both LDA and DTC achieved a test accuracy of 95.61% using their respective best models. The hyperparameter tuning process aimed to enhance the predictive capabilities of each classifier on the given dataset.

*Principle Component Analysis*  <br/>
Principle Component Analysis (PCA) suggested that the mean accuracy for logistic regression was poorer after PCA was performed than models without dimensional reduction. However, the PCA applied to the dataset resulted in a model with an accuracy of 93%. The classification report provided additional details on precision, recall, and F1-score for each class (0 and 1). The confusion matrix illustrated the distribution of correct and incorrect predictions, with 53 true negatives, 32 true positives, 2 false positives, and 4 false negatives. The overall performance metrics indicate a well-performing model after the application of PCA. 

### **Conclusions** <br/>
In the analysis of breast cancer diagnosis using machine learning models, SVM demonstrated the highest accuracy at 98%, followed closely by Logistic Regression and the Genetic Algorithm, both achieving 97% accuracy. Among linear models, Logistic Regression (97%) and Linear Discriminant Analysis (95%) performed exceptionally well, while ensemble models such as Random Forest (96%) and Decision Tree (95%) excelled in classification tasks. Notably, K-Nearest Neighbour, a non-parametric algorithm, had a slightly lower performance at 94%, although the accuracy remained at a satisfactory level.
Additionally, applying Principal Component Analysis (PCA) for dimensional reduction led to a reduction in the mean accuracy of Logistic Regression compared to the model without dimensionality reduction.
Key features influencing predictions were identified using various techniques. Recursive feature elimination and cross-validation with Logistic Regression highlighted radius, texture, perimeter, and area as the most predictive features. Random Forest classification emphasized area3, concave_points3, and concave_points1 using the extracted feature importance, while Logistic Regression pointed to texture3, radius2, symmetry3, and concave_points1 using the absolute coefficient values of the features. It's essential to note that Logistic Regression assumes a linear relationship, and interpretations are based on this assumption. In cases of non-linear relationships, more complex models may be considered.
SVM identified concave_point1, texture3, and radius2 as the most predictive features using feature coefficients from its linear kernel. Furthermore, the Genetic Algorithm selected top features such as radius1, perimeter1, and area1, followed by concavity1, symmetry1, perimeter2, area2, smoothness2, concavity2, concave_points2, fractal_dimension2, radius3, area3, smoothness3, and fractal_dimension3. This suggests that these features significantly contribute to the predictive performance of the model.

### **Recommendations** <br/>
1.	Logistic regression:
The logistic regression model exhibits excellent predictive capabilities, with high accuracy, precision, and recall values for both classes. It is recommended to further explore and fine-tune this model for potential deployment in clinical settings.

2.	Random Forest:
The Random Forest model demonstrates strong overall performance, balancing precision, recall, and F1-score for both benign and malignant cases. It is suggested to leverage the optimised hyperparameters obtained through the randomized search to enhance the model's efficiency.

3.	KNN & GB: 
KNN performs excellently with a high mean accuracy, suggesting its suitability for classification tasks. On the other hand, GB, while achieving a lower accuracy, could still be valuable in scenarios where interpretability and simplicity are prioritized. A future consideration would be the trade-offs between accuracy and model complexity when choosing between these two models.

4.	SVM:
The SVM model showcases outstanding accuracy, precision, and recall, particularly for classifying benign instances. This model holds promise for accurate identification of breast cancer cases and is recommended for further evaluation and potential deployment.

5.	Genetic Algorithm:
The genetic algorithm successfully achieved a perfect accuracy score, showcasing its effectiveness in feature selection. A consideration would be leveraging the insights from the selected features for interpretability and potentially improving model understanding.

6.	DTC & LDA:
Both LDA and DTC exhibit competitive test accuracies with optimized hyperparameters. Further investigation into their interpretability and generalization capabilities could guide their potential use in specific contexts.

7.	PCA:
The application of PCA resulted in a model with slightly reduced accuracy for logistic regression. In future, it would be important to carefully weigh the benefits of dimensionality reduction against potential information loss and model performance trade-offs.
In sum, the combination of models provides a diverse set of tools for breast cancer classification, each with its strengths and considerations. The choice of the most suitable model may depend on specific use-case requirements, interpretability needs, and the importance of different performance metrics. A thorough evaluation of these aspects will guide the selection and deployment of the most effective model for the given context in future.
 

### **References** <br/>
1.	Nallamala, S.H., Mishra, P. and Koneru, S.V., 2019. Breast cancer detection using machine learning way. Int J Recent Technol Eng, 8(2-3), pp.1402-1405.

2.	Ak, M.F., 2020, April. A comparative analysis of breast cancer detection and diagnosis using data visualization and machine learning applications. In Healthcare (Vol. 8, No. 2, p. 111). MDPI.

3.	Mohammed, S.A., Darrab, S., Noaman, S.A. and Saake, G., 2020. Analysis of breast cancer detection using different machine learning techniques. In Data Mining and Big Data: 5th International Conference, DMBD 2020, Belgrade, Serbia, July 14–20, 2020, Proceedings 5 (pp. 108-117). Springer Singapore.

4.	Bhise, S., Gadekar, S., Gaur, A.S., Bepari, S. and Deepmala Kale, D.S.A., 2021. Breast cancer detection using machine learning techniques. Int. J. Eng. Res. Technol, 10(7), pp.2278-0181.

5.	Hussain, L., Aziz, W., Saeed, S., Rathore, S. and Rafique, M., 2018, August. Automated breast cancer detection using machine learning techniques by extracting different feature extracting strategies. In 2018 17th IEEE International Conference On Trust, Security And Privacy In Computing And Communications/12th IEEE International Conference On Big Data Science And Engineering (TrustCom/BigDataSE) (pp. 327-331). IEEE.

6.	Pfob, A., Lu, S.-C. and Sidey-Gibbons, C. (2022) ‘Machine learning in medicine: A practical introduction to techniques for data pre-processing, hyperparameter tuning, and model comparison’, BMC Medical Research Methodology, 22(1). 

7.	Sidey-Gibbons, J.A. and Sidey-Gibbons, C.J., 2019. Machine learning in medicine: a practical introduction. BMC medical research methodology, 19, pp.1-18.
