import numpy as np

class PerceptronClassifier:
    '''Preceptron Binary Classifier uses Perceptron Learning Algorithm
       to do classification with two classes.

       Parameters
       ----------
       number_of_attributes : int
           The number of attributes of data set.

       Attributes
       ----------
       weights : list of float
           The list of weights corresponding &amp;lt;
           g class="gr_ gr_313 gr-alert gr_gramm gr_inline_cards gr_run_anim 
           Grammar multiReplace" id="313" data-gr-id="313"&amp;gt;with&amp;
           lt;/g&amp;gt; input attributes.

       errors_trend : list of int
           The number of misclassification for each training sample.
       '''
    def __init__(self, number_of_attributes: int):
        # Initialize the weigths to zero
        # The size is the number of attributes plus the bias, i.e. x_0 * w_0
        self.weights = np.zeros(number_of_attributes + 1)

        # Record of the number of misclassify for each train sample
        self.misclassify_record = []

        self._label_map = {}
        self._reversed_label_map = {}

    def _linear_combination(self, sample):
        '''linear combination of sample and weights'''
        return np.inner(sample, self.weights[1:])

    def train(self, samples, labels, max_iterator=10):
        '''Train the model

        Parameters
        ----------
        samples : two dimensions list
            Training data set
        labels : list of labels
            Class labels. The labels can be anything as long as it has 
                          only two types of labels.
        max_iterator : int
            The max iterator to stop the training process
            in case the training data is not converaged.
        '''
        # Build the label map to map the original labels to numerical labels
        # For example, ['a', 'b', 'c'] -&amp;gt; {0 : 'a', 1 : 'b', 2 : 'c'}
        self._label_map = {1 : list(set(labels))[0], -1 : list(set(labels))[1]}
        self._reversed_label_map = {value : key for key, value in self._label_map.items()}

        # Transfer the labels to numerical labels
        transfered_labels = [self._reversed_label_map[index] for index in labels]

        for _ in range(max_iterator):
            misclassifies = 0
            for sample, target in zip(samples, transfered_labels):
                linear_combination = self._linear_combination(sample)
                update = target - np.where(linear_combination &amp;gt;= 0.0, 1, -1)

                # use numpy.multiply to multiply element-wise
                self.weights[1:] += np.multiply(update, sample)
                self.weights[0] += update

                # record the number of misclassification
                misclassifies += int(update != 0.0)

            if misclassifies == 0:
                break
            self.misclassify_record.append(misclassifies)

    def classify(self, new_data):
        '''Classify the sample based on the trained weights

        Parameters
        ----------
        new_data : two dimensions list
            New data to be classified

        Return
        ------
        List of int
            The list of predicted class labels.
        '''
        predicted_result = np.where((self._linear_combination(new_data) + 
                                     self.weights[0]) &amp;gt;= 0.0, 1, -1)
        return [self._label_map[item] for item in predicted_result]
