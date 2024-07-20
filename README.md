# Data Touches Everything

## About

This repository serves as a comprehensive record of my academic journey and a platform for organizing my knowledge and goals in AI and Data Science as I pursue my Data Analystic and Data Science master's degree at Georgia Tech, guided by the [Artificial Intelligence Roadmap](https://i.am.ai/roadmap/#note). With a strong foundation in essential components that underlie modern data science methodologies, key areas include:

- **Optimization:** Fundamental techniques crucial for developing machine learning algorithms, sample code implemented using Python.
  
- **Machine Learning Models:** Focus on algorithms suited for handling large datasets with low-dimensional features, encompassing foundational tasks such as regression, classification, clustering, and dimensionality reduction. Advanced ensemble methods like boosting and bagging are also explored for enhanced model performance.
  
- **High-Dimensional Data Analysis:** Specialized methods for extracting informative features from datasets characterized by high dimensionality and limited samples. This includes advanced techniques in image processing, tensor analysis, and regularization.
  
- **Deep Learning:** Advanced modeling techniques designed for large-scale datasets and complex feature sets. Upcoming studies in Fall 2024 will focus on neural networks, gradient descent optimization, and advanced architectures such as convolutional and recurrent neural networks using PyTorch.




## Table of contents
- [About](#about)
- [Table of Contents](#table-of-contents)
- [Optimization](#optimization-course-at-georgia-tech)
- [Machine Learning](#machine-learning-course-at-georgia-tech)
- [High-Dimensional Data Analysis](#high-dimensional-data-analysis-course-at-georgia-tech)
- [Deep Learning (Fall 2024)](#deep-learning-course-at-georgia-tech-to-be-updated-in-fall-2024)
- [Additional Resources](#additional-resources)
- [Future Update](#future-update)

## Optimization Course at Georgia Tech
This graduate-level optimization course covered essential concepts, models, and algorithms in depth. Beginning with foundational principles and mathematical underpinnings, the curriculum progressed through linear optimization techniques, including advanced topics such as the simplex method and duality theory. Subsequent modules explored nonlinear and convex conic optimization, broadening understanding beyond linear models. Additionally, the course included a study of integer optimization, providing insights into integrating integer decision variables into optimization frameworks.

Practical skills were developed in formulating and solving complex optimization problems using Python-based tools. This experience enhanced proficiency in optimization theory, computational methods, and their applications in modern data analytics.

| Topics  | Description | Implementation <br> (Private)           | 
| :---: | :--- | :---: | 
| Convex & Nonconvex Optimization | Implementation of convex optimization using Newton's method and nonconvex optimization with Scipy's minimize function. | Python (numpy, scipy) |
| Linear Programming | Solution of common manufacturing production and electric power network problems using linear programming optimization techniques. | Python (numpy, cvxpy) |
| Solving Linear Program Using Basic Feasible Solutions | Identification of all possible basic solutions and determination of the optimal solution among them. | Python (numpy)|
|Solving Linear Program Using Simplex Method   ★ | Implementation of the simplex method from scratch to solve linear programming problems. | Python (numpy) |
| Cutting Stock Problem With Column Generation   ★ | Solution of the cutting stock problem using column generation, applied to solve the pricing problem. | Python (numpy, cvxpy)  |
| Location Optimization Using SOCP | Application of second-order cone programming to identify optimal locations for optimization problems. | Python (numpy, cvxpy)|


**Topic Learned:**
- Module 1: Introduction
- Module 2: Illustration of Optimization Process
- Module 3: Mathematical Concepts Review
- Module 4: Convexity
- Module 5: Outcomes of Optimization
- Module 6: Optimality Certificates
- Module 7-8: Unconstrained Optimization
- Module 9-10: Linear Optimization Modeling
- Module 11-12: Advanced Linear Optimization
- Module 13-14: Geometric and Algebraic Aspects of Linear Optimization
- Module 15-16: Simplex Method and Further Development
- Module 17-18: Advanced Optimization Techniques
- Module 19-20: Nonlinear Optimization Modeling
- Module 21-23: Convex and Conic Programming
- Module 24: Semi-Definite Programming
- Module 25-30: Discrete Optimization

**Additional Resources:**
- [Convex Optimization – Boyd and Vandenberghe   ★](https://stanford.edu/~boyd/cvxbook/)
- [ORF363 Computing and Optimization ](https://aaa.princeton.edu/orf363)
- [ORF523 Convex and Conic Optimization](https://aaa.princeton.edu/orf523)
 
## Machine Learning Course at Georgia Tech
This graduate-level machine learning course delved deeply into essential methodologies, theories, and algorithms across various domains. Beginning with fundamental concepts in clustering, dimensionality reduction, and statistical modeling, the curriculum progressed to advanced techniques such as Gaussian mixture models, support vector machines (SVM), and ensemble methods like AdaBoost and random forests. Practical application was a key focus, with hands-on projects using Python for algorithm implementation and analysis of real-world datasets. The course provided a thorough understanding of machine learning principles, equipping participants with the skills to tackle intricate analytical problems, optimize model performance, and derive meaningful insights from data.


| Topic                                | Description                                                                                           | Implementation    <br> (Private)                                                                                       |
|:-------------------------------------:|:-----------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------:|
| Clustering and K-means               | Image Compression using K means Algorithm  | Python (numpy) |
| Clustering and K-means               | Evaluation of K means and K median Clustering on MNIST Dataset |Python (numpy)|
| Spectral Clustering                  |  Political Blogosphere Analysis Using Spectral Clustering  [(Spectral Clustering, Ulrike von Luxburg)](https://arxiv.org/abs/0711.0189)                                           | Python (numpy) |
| Dimensionality Reduction and PCA     |  Eigenface Generation and Analysis using PCA on the Olivetti Faces Dataset                          | Python (numpy)|
| Nonlinear Dimensionality Reduction  ★ |   ISOMAP Algorithm Implementation and Visualization for Facial Image Analysis [(ISOMAP Paper)](https://www.science.org/doi/10.1126/science.290.5500.2319)                        | Python (numpy)|
| Density Estimation                   |  Analyzing Brain Structure and Categorical Labels Data Distribution                                  | Python (matplotlib, seaborn)|
| Gaussian Mixture Model and EM Algorithm |  Image Classification with Gaussian Mixture Models using EM                                         | Python (numpy)|
| Classification   ★                     |   Comparing ML Classifiers: Model Performance & Decision Boundaries                | Python (sklearn) |
| Anomaly Detection                    |  CUSUM for Distribution Shift Detection                                                             | Python (numpy) |
| Feature Selection & Random Forest    |  Fine Tuning Machine Learning Models for CART Random Forest and One Class SVM       [(Random Forests, Leo Breiman)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)                   | Python (numpy, sklearn) |
|  Nonlinear Regression                |  Locally Weighted Linear Regression with Bias Variance Tradeoff and Hyperparameter Fine Tuning       | Python (numpy) |
| Machine Learning Project  ★            |   Assessing Avocado Pricing Dynamics Utilizing Climate Transportation Cost and Macroeconomic Metrics in California. <br>Data Sources:    [ Federal Reserve Economic Data](https://fred.stlouisfed.org)   \| [National Weather Service](https://www.weather.gov/)   \| [U.S. Energy Information Administration (EIA)](https://www.eia.gov/)  | Python (matplotlib, seaborn, numpy, sklearn)|




**Topic Learned:**
- Module 1: Clustering and K-means
- Module 2: Spectral Clustering
- Module 3: Dimensionality Reduction and PCA
- Module 4: Nonlinear Dimensionality Reduction
- Module 5: Density Estimation
- Module 6: Gaussian Mixture Model and EM Algorithm
- Module 7: Basics of Optimization Theory
- Module 8: Classification: Naïve Bayes and Logistic Regression
- Module 9: Support Vector Machine (SVM), Neural Networks
- Module 10: Feature Selection
- Module 11: Anomaly Detection
- Module 12: Boosting Algorithms and AdaBoost
- Module 13: Random Forest
- Module 14: Bias-Variance Tradeoff, Cross-Validation, Kernel Methods, Reinforcement Learning
- Module 15: Intro to Reinforcement Learning


**Books:**
- [ The Elements of Statistical Learning  ★](https://www.google.com/books/edition/The_Elements_of_Statistical_Learning/yPfZBwAAQBAJ?hl=en)
- [ A Mathematical Introduction to Data Science by Yuan Yao  ★](https://www.math.pku.edu.cn/teachers/yaoy/reference/book05.pdf)
- [Pattern Recognition and Machine Learning](https://www.google.com/books/edition/Pattern_Recognition_and_Machine_Learning/kOXDtAEACAAJ?hl=en)
- [Foundations of Machine Learning](https://www.google.com/books/edition/Foundations_of_Machine_Learning_second_e/V2B9DwAAQBAJ?hl=en)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
- [Collaborative filtering  (recommender systems)](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Fairness and machine learning](https://fairmlbook.org/)
  
**Additional Resources:**
- [ Intuition for the Algorithms of Machine Learning — Cynthia Rudin  ★](https://users.cs.duke.edu/~cynthia/teaching.html)
- [ Statistical Machine Learning — Ulrike von Luxburg  ★](https://www.youtube.com/playlist?list=PL05umP7R6ij2XCvrRzLokX6EoHWaGA2cC)
- [Mathematics for Machine Learning — Ulrike von Luxburg](https://www.youtube.com/playlist?list=PL05umP7R6ij1a6KdEy8PVE9zoCv6SlHRS)
- [Statistical Learning with Python](https://www.youtube.com/playlist?list=PLoROMvodv4rPP6braWoRt5UCXYZ71GZIQ)
- [CS229: Machine Learning — Andrew Ng](https://cs229.stanford.edu/)

  
## High-Dimensional Data Analysis Course at Georgia Tech
This graduate-level course in high-dimensional data analysis is designed to equip professionals with essential skills in feature extraction and dimensionality reduction in statistical machine learning. The curriculum focuses on key topics such as functional data analysis, advanced image processing techniques, multilinear algebra, tensor analysis, and advanced regularization techniques for handling complex datasets. These foundational areas provide vital tools for effectively understanding and interpreting data, improving the extraction of meaningful features, and simplifying data analysis. Understanding these techniques prepares professionals to address challenges in image analysis, optimize predictive models to prevent errors, and gain useful insights across various data-driven fields.
| Topic                                | Additional Readings                                                                                           | Implementation <br>  (To Be Updated)                                                                                       |
|:-------------------------------------:|:-----------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------:|
| Linear Regression     <br>    Splines         <br>   BSplines       <br>    Smoothing Splines     <br>  Kernel Smoothers    <br>  FPCA        | [1. The Elements of Statistical Learning Page 43-52](https://www.google.com/books/edition/The_Elements_of_Statistical_Learning/yPfZBwAAQBAJ?hl=en)    <br>  [2. The Elements of Statistical Learning Page 139-144](https://www.google.com/books/edition/The_Elements_of_Statistical_Learning/yPfZBwAAQBAJ?hl=en)  <br>    [3. The Elements of Statistical Learning Page 186-189](https://www.google.com/books/edition/The_Elements_of_Statistical_Learning/yPfZBwAAQBAJ?hl=en) <br>   [4. The Elements of Statistical Learning Page 144-161](https://www.google.com/books/edition/The_Elements_of_Statistical_Learning/yPfZBwAAQBAJ?hl=en)  <br> [5. The Elements of Statistical Learning Page 191-208](https://www.google.com/books/edition/The_Elements_of_Statistical_Learning/yPfZBwAAQBAJ?hl=en)   <br> [6. Functional Data Analysis for Sparse Longitudinal Data](https://www.researchgate.net/publication/4741968_Functional_Data_Analysis_for_Sparse_Longitudinal_Data)   | [Python (numpy)] |
|  Image Filtering and Convolution    <br>      Image Transformation  & Edge Detection  <br>   Image segmentation       | [1. A Concise Introduction to Image Processing Using C++ Chapter 2](https://www.google.com/books/edition/A_Concise_Introduction_to_Image_Processi/fp7SBQAAQBAJ?hl=en)   <br>  [2. A Concise Introduction to Image Processing Using C++ Chapter 3](https://www.google.com/books/edition/A_Concise_Introduction_to_Image_Processi/fp7SBQAAQBAJ?hl=en) <br>  [3. A Concise Introduction to Image Processing Using C++ Chapter 4](https://www.google.com/books/edition/A_Concise_Introduction_to_Image_Processi/fp7SBQAAQBAJ?hl=en)| [Python (numpy)] |
| Tensor Preliminaries      <br>  Tensor Decomposition      <br>  Tensor  Applications I      <br>   Tensor  Applications II      <br>    Tensor  Applications III     | [1. Tensor Decompositions and Applications 455-462](https://www.jstor.org/stable/25662308?seq=1#metadata_info_tab_contents)  <br>   [2. Tensor Decompositions and Applications 462-480](https://www.jstor.org/stable/25662308?seq=1#metadata_info_tab_contents) <br>  [3. Structured Point Cloud Data Analysis via Regularized Tensor Regression for Process Modeling and Optimization](https://arxiv.org/abs/1807.10278)<br>  [4. Image-Based Prognostics Using Penalized Tensor Regression](https://arxiv.org/abs/1706.03423)  <br>  [5. Tensor decompositions for feature extraction and classification of high dimensional datasets](https://www.researchgate.net/publication/228553771_Tensor_decompositions_for_feature_extraction_and_classification_of_high_dimensional_datasets)   | [Python (numpy)] |
| Optimization:  First order methods   <br>   Optimization:  Second order methods   | [1. Convex Optimization – Boyd and Vandenberghe Pages 1-11, 21-35, 67-79, 457 - 475 and 484-487 ](https://stanford.edu/~boyd/cvxbook/)  <br>   [2. Numerical Methods for Least Squares Problems Ch 9](https://www.google.com/books/edition/Numerical_Methods_for_Least_Squares_Prob/aQD1LLYz6tkC?hl=en)  | [Python (numpy)] |
| Proximal Gradient Descent   <br>        Coordinate Descent <br>       ALM and ADMM    | [1. Optimization for Machine Learning Page 27-32](https://www.google.com/books/edition/Optimization_for_Machine_Learning/JPQx7s2L1A8C?hl=en)  <br>   [2. Optimization for Machine Learning Page 32-34](https://www.google.com/books/edition/Optimization_for_Machine_Learning/JPQx7s2L1A8C?hl=en)   <br>   [3. Distributed Optimization and Statistical Learning Via the Alternating Direction Method of Multipliers Page 1-24](https://www.google.com/books/edition/Distributed_Optimization_and_Statistical/8MjgLpJ0_4YC?hl=en) | [Python (numpy)] |
| Ridge Regression     &LASSO     <br>   NNG       <br>   Adaptive LASSO     <br>   Grouped LASSO      <br>    Elastic Net          | [1. The Elements of Statistical Learning Page 61-73](https://www.google.com/books/edition/The_Elements_of_Statistical_Learning/yPfZBwAAQBAJ?hl=en)  <br>   [2. The adaptive LASSO ad its oracle properties](https://www.researchgate.net/publication/4742238_The_adaptive_LASSO_ad_its_oracle_properties)    <br>   [3. Better Subset Regression Using the Nonnegative Garrote](https://www.researchgate.net/publication/243776325_Better_Subset_Regression_Using_the_Nonnegative_Garrote)   <br>   [4. Model Selection and Estimation in Regression With Grouped Variables](https://www.researchgate.net/publication/4993325_Model_Selection_and_Estimation_in_Regression_With_Grouped_Variables)    <br>  [5. Regularization and variable selection via the elastic net](https://www.researchgate.net/publication/227604843_Zou_H_Hastie_T_Regularization_and_variable_selection_via_the_elastic_net_J_R_Statist_Soc_B_2005672301-20)  | [Python (numpy)] |
| Compressive Sensing     <br>      Matrix Completion      <br>   Robust PCA       <br>   Smooth-Sparse Decomposition    <br>   RKHS Ridge Kernel Regresion      | [1. An introduction to compressive sampling](https://www.researchgate.net/publication/3322018_Wakin_MB_An_introduction_to_compressive_sampling_IEEE_Signal_Process_Mag_252_21-30)  <br>   [2. A Singular Value Thresholding Algorithm for Matrix Completion](https://www.researchgate.net/publication/220133672_A_Singular_Value_Thresholding_Algorithm_for_Matrix_Completion)  <br>   [3. Exact Matrix Completion via Convex Optimization](https://www.researchgate.net/publication/227101649_Exact_Matrix_Completion_via_Convex_Optimization)   <br>   [4.  Robust Principal Component Analysis: Exact Recovery of Corrupted Low-Rank Matrices via Convex Optimization](https://www.researchgate.net/publication/221618324_Robust_Principal_Component_Analysis_Exact_Recovery_of_Corrupted_Low-Rank_Matrices_via_Convex_Optimization)   <br>   [5.  Robust Principal Component Analysis?](https://arxiv.org/abs/0912.3599)  <br>    [6. Introduction to RKHS, and some simple kernel algorithms](https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf)   <br>   [7. Anomaly Detection in Images with Smooth Background Via Smooth-Sparse Decomposition](https://www.researchgate.net/publication/283520589_Anomaly_Detection_in_Images_with_Smooth_Background_Via_Smooth-Sparse_Decomposition)      | [Python (numpy)] |




**Topic Learned:**
- Module 1: Functional Data Analysis
  - Linear Regression, Splines, BSplines , Smoothing Splines, Kernel Smoothers, FPCA

- Module 2: Image Analysis
  - Image Filtering and Convolution, Image Transformation & Edge Detection, Image segmentation

- Module 3: Tensor Data Analysis
  - Tensor Preliminaries, Tensor Decomposition, Tensor Applications 

- Module 4: Optimization I
  - First order methods, Second order methods

- Module 5: Optimization II
  - Proximal Gradient Descent, Coordinate Descent, ALM and ADMM

- Module 6: Regularization
  - Ridge Regression & LASSO, Adaptive LASSO, Grouped LASSO, Elastic Net

- Module 7: Regularization Applications
  - Compressive Sensing, Matrix Completion, Robust PCA,
  - Smooth-Sparse Decomposition, RKHS Ridge Kernel Regression



**Additional Resources:**
- [ High-Dimensional Data Analysis with Low-Dimensional Models — John Wright, Yi Ma  ★](https://book-wright-ma.github.io/)


## Deep Learning Course at Georgia Tech (To be updated in Fall 2024)
Deep learning, a specialized branch of machine learning, focuses on extracting intricate hierarchical representations from raw data. Central to this field is artificial neural networks, which have revolutionized data processing across various domains such as image analysis, natural language processing, and decision-making tasks. This course delves into fundamental principles, mathematical foundations, and practical implementation of deep learning. Topics include optimization techniques like gradient descent and backpropagation, foundational neural network modules such as linear and convolutional layers, and advanced architectures like recurrent neural networks and convolutional neural networks. Through hands-on programming assignments using PyTorch, participants will learn to construct and optimize neural networks, apply them to real-world applications, choose appropriate models for diverse problems, and understand ongoing research challenges in the field.


**Additional Resources:**
- [ Attention Is All You Need ★](https://arxiv.org/abs/1706.03762)
- [Dive into Deep Learning  ](https://d2l.ai/)

  
## Additional Resources
- [AI-Expert-Roadmap ★](https://github.com/AMAI-GmbH/AI-Expert-Roadmap)
- [Trustworthy Online Controlled Experiments (A/B Testing) ★](https://www.google.com/books/edition/Trustworthy_Online_Controlled_Experiment/NHjQDwAAQBAJ?hl=en)
- [Github Ranking ](https://github.com/EvanLi/Github-Ranking)
- [Kaggle Competition ](https://www.kaggle.com/competitions)
- [Algorithms and Data Structures using Python ](https://runestone.academy/ns/books/published/pythonds3/index.html)
- [Machine Learning & Causal Inference](https://www.youtube.com/playlist?list=PLxq_lXOUlvQAoWZEqhRqHNezS30lI49G-)
- [Probabilistic Machine Learning — Philipp Hennig](https://www.youtube.com/playlist?list=PL05umP7R6ij1tHaOFY96m5uX3J21a6yNd)
- [Lil'Log](https://lilianweng.github.io/)


## Future Update
This repository will be continually updated as I progress in my studies and research. A new reinforcement learning section will be particularly expanded in 2025, with detailed explanations and practical implementations of various reinforcement learning techniques.
