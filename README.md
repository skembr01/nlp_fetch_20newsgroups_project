# Unsupervised Learning NLP

The data used in this project comes from scikit-learn's dataset and is the fetch_20newsgroups. This data contains approximately 18000 posts from 20 different news groups (these are essentially like the precursor to Reddit threads and were around in the 1980-1990s). The data contains the text of each post and the asociated thread it was posted into. Thus, this data provides us with a natural language processing (NLP) problem, with a particular emphasis on topic modeling.

When working with text, there oft needs to be cleaning in order for the models to work with the data. I began this by creating two functions to clean the posts. Firstly, I utilized a function which converts the whole post into lower case and which removes punctuation as determined by the string package. After applying this function to both training and test sets, I utlized my second function to remove stop words. Stop words are those which are quite common in the english value and often provide little signficance to the overall meaning of a text document. An example of these would be 'a', 'these', and 'so'. By removing these, the models can work on words which would have more import to the overall meaning.

After this, I then utilized the method of topic frequency-inverse document frequency. The mathematics of this is:

term_frequency(tf) * inverse document frequency(idf),

    tf = # of times word in document / # of terms in document
    
    idf = log (# of documents / # of documents containing term)
    
This method is utilized to measure how important a word is to the overall document it is obtained from. The more times the word appears in a document, the more this importance grows (hence, we can see the importance of removing stopwords). In order to perform this I utilized scikit-learn to fit on the X_train and transform it and then transforming X_test. When this was completed, an entire vocabulary of 138,397 words were used. These methods also produced the data into sparse matrices as the large data may can lead to excessive runtimes.

The first unsupervised method I wanted to use was NMF, which I did with scikit-learn. This factorization breaks the initial sparse matrix into W and H matrices which when multiplied together form a matrix very similar to the original data. As I knew the number of topics a priori, I was able to set the number of topics for the factorization.

Lastly, in terms of NMF, I outputted the beta-divergence and the weights associated with each component in the model. The beta-divergence is the difference between WH and the intitial data matrix. As NMF does not provided an exact factorization, maintaining a low beta-divergence indicates a well-performing model. In this case, a beta-divergence of 104 was obtained, which is an adequate result given the number of topics. Lastly, I charted the weights associated with each component in the H matrix and in the model.

After, NMF I wanted to utilize another decomposition method of svd. This method produces an exact decomposition of the intitial data matrix. This results in higher precision among the mathematics; however, it also takes much longer to perform and is much more computationally expensive. Due to this, I utlized a randomized_svd method from scikit-learn. This method has reduced accuracy and precision, but allowed for manageability on my personal computer. Instead of decomposing the entire data matrix, this method randomly utlizes a part of the intitial data matrix, giving its performance changes.
