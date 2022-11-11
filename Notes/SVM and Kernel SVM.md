# Summary

In this article, you will learn about **SVM** or **Support Vector Machine**, which is one of the most popular AI algorithms (it’s one of the top 10 AI algorithms) and about the **Kernel Trick**, which deals with **non-linearity** and **higher dimensions**. We will touch topics like **hyperplanes, Lagrange Multipliers**, we will have **visual examples** and **code examples** (similar to the code example used in the [KNN chapter](http://dummyprogramming.com/stupid-simple-ai-series/)) to better understand this very important algorithm.

# SVM Explained

The **Support Vector Machine** is a **supervised learning algorithm** mostly used for **classification** but it can be used also for **regression**. The main idea is that based on the labeled data (training data) the algorithm tries to find the **optimal hyperplane** which can be used to classify new data points. In two dimensions the hyperplane is a simple line.

**Usually** a learning algorithm tries to learn the **most common characteristics (what differentiates one class from another)** of a class and the classification is based on those representative characteristics learnt (so classification is based on differences between classes). The **SVM** works in the other way around. It **finds** the **most similar examples** between classes. Those will be the **support vectors**.

As an example, lets consider two classes, apples and lemons.

Other algorithms will learn the most evident, most representative characteristics of apples and lemons, like apples are green and rounded while lemons are yellow and have elliptic form.

In contrast, SVM will search for apples that are very similar to lemons, for example apples which are yellow and have elliptic form. This will be a support vector. The other support vector will be a lemon similar to an apple (green and rounded). So **other algorithms** learns the **differences** while **SVM** learns **similarities**.

If we visualize the example above in 2D, we will have something like this:

![](https://miro.medium.com/max/700/0*eH4sQUkJc8YT76jL.png)

As we go from left to right, all the examples will be classified as apples until we reach the yellow apple. From this point, the confidence that a new example is an apple drops while the lemon class confidence increases. When the lemon class confidence becomes greater than the apple class confidence, the new examples will be classified as lemons (somewhere between the yellow apple and the green lemon).

Based on these support vectors, the algorithm tries to find **the best hyperplane that separates the classes**. In 2D the hyperplane is a line, so it would look like this:

![](https://miro.medium.com/max/700/0*6f4wnyOzFom45hh9.png)

Ok, but **why did I draw the blue boundary like in the picture** **above?** I could also draw boundaries like this:

![](https://miro.medium.com/max/700/0*jfUOIfBTU6oRyaRv.png)

As you can see, we have **an infinite number of possibilities to draw the decision boundary**. So how can we find the optimal one?

# Finding the Optimal Hyperplane

Intuitively the **best line** is the line that is **far away from both apple and lemon examples** (has the largest margin). To have optimal solution, we have to **maximize the margin in both ways** (if we have multiple classes, then we have to maximize it considering each of the classes).

![](https://miro.medium.com/max/700/0*VpDK7t9et977TSwG.png)

So if we compare the picture above with the picture below, we can easily observe, that the first is the optimal hyperplane (line) and the second is a sub-optimal solution, because the margin is far shorter.

![](https://miro.medium.com/max/700/0*6jEF9zPYt7v650GQ.png)

Because we want to maximize the margins taking in consideration **all the classes**, instead of using one margin for each class, **we use a “global” margin**, **which takes in consideration all the classes**. This margin would look like the purple line in the following picture:

![](https://miro.medium.com/max/700/0*SgZeIjb1kKpKyk7I.png)

This margin is **orthogonal** to the boundary and **equidistant** to the support vectors.

So where do we have vectors? Each of the calculations (calculate distance and optimal hyperplanes) are made in **vectorial space**, so each data point is considered a vector. The **dimension** of the space **is defined by the number of attributes** of the examples. To understand the math behind, please read this brief mathematical description of vectors, hyperplanes and optimizations: [SVM Succintly](https://www.svm-tutorial.com/).

All in all, **support vectors** are data points that **defines the position and the margin of the hyperplane**. We call them **“support” vectors**, because these are the representative data points of the classes, **if we move one of them, the position and/or the margin will change**. Moving other data points won’t have effect over the margin or the position of the hyperplane.

To make classifications, we don’t need all the training data points (like in the case of KNN), we have to save only the support vectors. In worst case all the points will be support vectors, but this is very rare and if it happens, then you should check your model for errors or bugs.

So basically the **learning is equivalent with finding the hyperplane with the best margin**, so it is a simple **optimization problem**.

# Basic Steps

The basic steps of the SVM are:

1.  select **two hyperplanes** (in 2D) which separates the data **with no points between them** (red lines)
2.  **maximize their distance** (the margin)
3.  the **average line** (here the line half way between the two red lines) will be the **decision boundary**

This is very nice and easy, but finding the best margin, the optimization problem is not trivial (it is easy in 2D, when we have only two attributes, but what if we have N dimensions with N a very big number)

To solve the optimization problem, we use the **Lagrange Multipliers**. To understand this technique you can read the following two articles: [Duality Langrange Multiplier](https://www.svm-tutorial.com/2016/09/duality-lagrange-multipliers/) and [A Simple Explanation of Why Langrange Multipliers Wroks](https://medium.com/@andrew.chamberlain/a-simple-explanation-of-why-lagrange-multipliers-works-253e2cdcbf74).

Until now we had linearly separable data, so we could use a line as class boundary. But what if we have to deal with non-linear data sets?

# SVM for Non-Linear Data Sets

An example of non-linear data is:

![](https://miro.medium.com/max/700/0*3jWNwLMhrhazDmdg.png)

In this case **we cannot find a straight line** to separate apples from lemons. So how can we solve this problem. We will use the **Kernel Trick!**

The basic idea is that when a data set is inseparable in the current dimensions, **add another dimension**, maybe that way the data will be separable. Just think about it, the example above is in 2D and it is inseparable, but maybe in 3D there is a gap between the apples and the lemons, maybe there is a level difference, so lemons are on level one and apples are on level two. In this case, we can easily draw a separating hyperplane (in 3D a hyperplane is a plane) between level 1 and 2.

# Mapping to Higher Dimensions

To solve this problem we **shouldn’t just blindly add another dimension**, we should transform the space so we generate this level difference intentionally.

# Mapping from 2D to 3D

Let's assume that we add another dimension called **X3**. Another important transformation is that in the new dimension the points are organized using this formula **x1² + x2²**.

If we plot the plane defined by the **x² + y²** formula, we will get something like this:

![](https://miro.medium.com/max/517/0*4tGRdSHgOoKZoQAT.png)

Now we have to map the apples and lemons (which are just simple points) to this new space. Think about it carefully, what did we do? We just used a transformation in which **we added levels based on distance**. If you are in the origin, then the points will be on the lowest level. As we move away from the origin, it means that we are **climbing the hill** (moving from the center of the plane towards the margins) so the level of the points will be higher. Now if we consider that the origin is the lemon from the center, we will have something like this:

![](https://miro.medium.com/max/700/0*GKpdwmJ-YoZVWqeS.png)

Now we can easily separate the two classes. These transformations are called **kernels**. Popular kernels are: **Polynomial Kernel, Gaussian Kernel, Radial Basis Function (RBF), Laplace RBF Kernel, Sigmoid Kernel, Anove RBF Kernel**, etc (see [Kernel Functions](https://data-flair.training/blogs/svm-kernel-functions/) or a more detailed description [Machine Learning Kernels](https://mlkernels.readthedocs.io/en/latest/kernels.html)).

# Mapping from 1D to 2D

Another, easier example in 2D would be:

![](https://miro.medium.com/max/700/0*fMVjaxA1buUoH62L.png)

After using the kernel and after all the transformations we will get:

![](https://miro.medium.com/max/700/0*gpNYO_ZXmrt_Hjxg.png)

So after the transformation, we can easily delimit the two classes using just a single line.

In real life applications we won’t have a simple straight line, but we will have lots of curves and high dimensions. In some cases we won’t have two hyperplanes which separates the data with no points between them, so **we need some trade-offs, tolerance for outliers**. Fortunately the SVM algorithm has a so-called **regularization parameter** to configure the trade-off and to tolerate outliers.

# Tuning Parameters

As we saw in the previous section **choosing the right kernel is crucial**, because if the transformation is incorrect, then the model can have very poor results. As a rule of thumb, **always check if you have linear data** and in that case always **use linear SVM** (linear kernel). **Linear SVM is a parametric model**, but an **RBF kernel SVM isn’t**, so the complexity of the latter grows with the size of the training set. Not only is **more expensive to train an RBF kernel SVM**, but you also have to **keep the kernel matrix around**, and the **projection** **into** this “infinite” **higher dimensional space** where the data becomes linearly separable is **more expensive** as well during prediction. Furthermore, you have **more hyperparameters to tune**, so model selection is more expensive as well! And finally, it’s much **easier to overfit** a complex model!

# Regularization

The **Regularization Parameter** (**in python it’s called** **C**) tells the SVM optimization **how much you want to avoid miss classifying** each training example.

If the **C is** **higher**, the optimization will choose **smaller margin** hyperplane, so training data **miss classification rate will be lower**.

On the other hand, if the **C is** **low**, then the **margin will be big**, even if there **will be miss classified** training data examples. This is shown in the following two diagrams:

![](https://miro.medium.com/max/700/0*rvt2H-wO55hKjJ5Y.png)

As you can see in the image, when the C is low, the margin is higher (so implicitly we don’t have so many curves, the line doesn’t strictly follows the data points) even if two apples were classified as lemons. When the C is high, the boundary is full of curves and all the training data was classified correctly. **Don’t forget**, even if all the training data was correctly classified, this doesn’t mean that increasing the C will always increase the precision (because of overfitting).

# Gamma

The next important parameter is **Gamma**. The gamma parameter defines **how far the influence of a single training example reaches**. This means that **high Gamma** will consider only points **close** to the plausible hyperplane and **low** **Gamma** will consider **points at greater distance**.

![](https://miro.medium.com/max/700/0*P5cqyr_n84SQDuAN.png)

As you can see, decreasing the Gamma will result that finding the correct hyperplane will consider points at greater distances so more and more points will be used (green lines indicates which points were considered when finding the optimal hyperplane).

# Margin

The last parameter is the **margin**. We’ve already talked about margin, **higher margin results better model**, so better classification (or prediction). The margin should be always **maximized**.

# SVM Example using Python

In this example we will use the Social_Networks_Ads.csv file, the same file as we used in the previous article, see [KNN example using Python](http://dummyprogramming.com/stupid-simple-ai-series-knn/#knn-code-example).

In this example I will write down only the **differences** between SVM and KNN, because I don’t want to repeat myself in each article! If you want the **whole explanation** about how can we read the data set, how do we parse and split our data or how can we evaluate or plot the decision boundaries, then please **read the code example from the previous chapter** ([KNN](http://dummyprogramming.com/stupid-simple-ai-series-knn/#knn-code-example))!

Because the **sklearn** library is a very well written and useful Python library, we don’t have too much code to change. The only difference is that we have to import the **SVC** class (SVC = SVM in sklearn) from **sklearn.svm** instead of the KNeighborsClassifier class from sklearn.neighbors.

# Fitting SVM to the Training set  
from sklearn.svm import SVC  
classifier = SVC(kernel = 'rbf', C = 0.1, gamma = 0.1)  
classifier.fit(X_train, y_train)

After importing the SVC, we can create our new model using the predefined constructor. This constructor has many parameters, but I will describe only the most important ones, most of the time you won’t use other parameters.

The most important parameters are:

1.  **kernel:** the kernel type to be used. The most common kernels are **rbf** (this is the default value), **poly** or **sigmoid**, but you can also create your own kernel.
2.  **C:** this is the **regularization parameter** described in the [Tuning Parameters](https://towardsdatascience.com/svm-and-kernel-svm-fed02bef1200#tuning) section
3.  **gamma:** this was also described in the [Tuning Parameters](https://towardsdatascience.com/svm-and-kernel-svm-fed02bef1200#tuning) section
4.  **degree:** it is used **only if the chosen kernel is poly** and sets the degree of the polinom
5.  **probability:** this is a boolean parameter and if it’s true, then the model will return for each prediction, the vector of probabilities of belonging to each class of the response variable. So basically it will give you the **confidences for each prediction**.
6.  **shrinking:** this shows whether or not you want a **shrinking heuristic** used in your optimization of the SVM, which is used in [Sequential Minimal Optimization](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf). It’s default value is true, an **if you don’t have a good reason, please don’t change this value to false**, because shrinking will **greatly** **improve your performance**, for very **little loss** in terms of **accuracy** in most cases.

Now lets see the output of running this code. The decision boundary for the training set looks like this:

![](https://miro.medium.com/max/574/0*AiUdfzW8iJmkAGAi.png)

As we can see and as we’ve learnt in the [Tuning Parameters](https://towardsdatascience.com/svm-and-kernel-svm-fed02bef1200#tuning) section, because the C has a small value (0.1) the decision boundary is smooth.

Now if we increase the C from 0.1 to 100 we will have more curves in the decision boundary:

![](https://miro.medium.com/max/581/0*pl5Eu6AZI0s3SEym.png)

What would happen if we use C=0.1 but now we increase Gamma from 0.1 to 10? Lets see!

![](https://miro.medium.com/max/585/0*GOxuLqcSpXyhNNEZ.png)

What happened here? Why do we have such a bad model? As you’ve seen in the [Tuning Parameters](https://towardsdatascience.com/svm-and-kernel-svm-fed02bef1200#tuning) section, **high gamma** means that when calculating the plausible hyperplane we consider **only points which are close**. Now because the **density** of the green points **is high only in the selected green region**, in that region the points are close enough to the plausible hyperplane, so those hyperplanes were chosen. Be careful with the gamma parameter, because this can have a very bad influence over the results of your model if you set it to a very high value (what is a “very high value” depends on the density of the data points).

For this example the best values for C and Gamma are 1.0 and 1.0. Now if we run our model on the test set we will get the following diagram:

![](https://miro.medium.com/max/568/0*lJXOBji5rAlv7s1J.png)

And the **Confusion Matrix** looks like this:

![](https://miro.medium.com/max/201/0*5uVGfqbMbt9f-1VY.png)

As you can see, we’ve got only **3 False Positives** and only **4 False Negatives**. The **Accuracy** of this model is **93%** which is a really good result, we obtained a better score than using [KNN](http://dummyprogramming.com/stupid-simple-ai-series-knn/) (which had an accuracy of 80%).

**NOTE:** accuracy is not the only metric used in ML and also **not the best metric to evaluate a model**, because of the [Accuracy Paradox](https://towardsdatascience.com/accuracy-paradox-897a69e2dd9b). We use this metric for simplicity, but later, in the chapter **Metrics to Evaluate AI Algorithms** we will talk about the **Accuracy Paradox** and I will show other very popular metrics used in this field.

# Conclusions

In this article we’ve seen a very popular and powerful supervised learning algorithm, the **Support Vector Machine**. We’ve learnt the **basic idea**, what is a **hyperplane**, what are **support vectors** and why are they so important. We’ve also seen lots of **visual representations**, which helped us to better understand all the concepts.

Another important topic that we touched is the **Kernel Trick**, which helped us to **solve non-linear problems**.

To have a better model, we saw techniques to **tune the algorithm**. At the end of the article we had a **code example in Python**, which showed us how can we use the KNN algorithm.

[

![](https://miro.medium.com/max/700/1*tYE4n1ydG0AvXMn_-ADaXw@2x.png)

](https://ko-fi.com/zozoczako)

**I really like coffee, because it gives me the energy to write more articles.**

**If you liked this article, then you can show your appreciation and support by buying me a coffee!**

As final thoughts, I would like to give some **pros & cons** and some popular **use cases**.