import math
from sklearn.model_selection import KFold
from sklearn import metrics
import pandas as pd
import re

# Name: David Irwin
# Student Number: R00109532
# Word Occurrence: 300
# This code is unoptimized and is slow to run.


# Task 1
def Task1():
    # Read in excel file using pandas
    movie_reviews = pd.read_excel("movie_reviews.xlsx", engine='openpyxl')

    # Seperate it into four different series'. Condition is determined by its test/train coloumn
    training_data = movie_reviews.loc[movie_reviews["Split"].str.contains("train", case=False), "Review"]
    training_labels = movie_reviews.loc[movie_reviews["Split"] == "train", "Sentiment"]
    test_data = movie_reviews.loc[movie_reviews["Split"] == "test", "Review"]
    test_labels = movie_reviews.loc[movie_reviews["Split"] == "test", "Sentiment"]

    # print a count of how many positive & negative reviews are in the training & test data
    print("Training Set positive reviews:", len(training_labels[training_labels == "positive"]))
    print("Training Set negative reviews:", len(training_labels[training_labels == "negative"]))
    print("Test Set positive reviews:", len(test_labels[test_labels == "positive"]))
    print("Test Set negative reviews:", len(test_labels[test_labels == "negative"]), '\n')

    return training_data, training_labels, test_data, test_labels



# Task2
def Task2(training_data, min_word_length, min_word_occurrence):
    # Convert training series into a list
    training_data = training_data.tolist()

    # # replace everything that is not alphanumeric or a whitespace using regex, with whitespace. I replace with a
    # whitespace here because it joined words together when i used ''
    training_data = [re.sub('[^a-zA-Z ]', ' ', i) for i in training_data]
    training_data = [i.lower() for i in training_data]  # Convert everything to lowercase
    training_data = [i.split() for i in training_data]  # Split everything into separate words

    # This loop gets rid of sublists (flattens nested list) and only appends words to the flattened list where the length
    # is greater than the specified value (given in the function parameter)
    training_data_no_sublist = []
    for lists in training_data:
        for word in lists:
            if len(word) >= min_word_length:
                training_data_no_sublist.append(word)


    training_data_no_sublist = pd.Series(training_data_no_sublist)  # Converts list to Series

    # This removes words from series where the word is less than the occurrence value (specified in parameter of function)
    training_data_no_sublist = training_data_no_sublist.value_counts()[
        training_data_no_sublist.value_counts() >= min_word_occurrence]

    # Get the index values of the training data series as it contains the word values. Convert this series into list
    words = pd.Series(
        training_data_no_sublist.index).tolist()

    return words  # list of words with minimum length and minimum occurrence


def Task3(training_data, training_labels, word_list):

    # convert training data & labels into lists
    training_data = training_data.tolist()
    training_labels = training_labels.tolist()

    # Convert data and labels into a dataframe
    dataframe = pd.DataFrame({"Training_Data": training_data, "Training_Labels": training_labels})

    # Split dataframe into a positive dataframe & negative dataframe
    training_set_positive = dataframe['Training_Data'][dataframe['Training_Labels'] == 'positive']
    training_set_negative = dataframe['Training_Data'][dataframe['Training_Labels'] == 'negative']

    # convert dataframe to list
    training_set_positive = training_set_positive.tolist()
    training_set_negative = training_set_negative.tolist()

    dict_of_positive_reviews, dict_of_negative_reviews = {}, {} # create two empty dictionaries
    # ------------------------------------------------------------------------------------------------------------------------------------------

    # First I deal with the training_set_positive list
    # replace everything that is not alphanumeric or a whitespace using regex. I replace with a whitespace here because
    # it joined words together when i used ''
    training_set_positive = [re.sub("'", '', i) for i in
                             training_set_positive]
    training_set_positive = [re.sub('[^a-zA-Z ]', ' ', i) for i in
                             training_set_positive]
    training_set_positive = [i.lower() for i in training_set_positive]  # Convert everything to lowercase

    # This block of of code is for getting the total number of words in the positive training dataset after they are split
    training_data_test = [i.split() for i in training_set_positive]  # Split everything into separate words
    training_data_no_sublist = []  # This gets rid of sublists and creates a list where words are longer than specified length
    for lists in training_data_test:
        for word in lists:
            training_data_no_sublist.append(word)
    df_test = pd.DataFrame(training_data_no_sublist)
    # This value is then used in Task 5
    total_words_in_positive_reviews = len(df_test)

    # Here I search through the positive dataset, count each occurrence of the words specified in the word list
    # I add the final count of each word occurrence and add it to the dictionary with the word value
    count = 0
    for word in range(0, len(word_list)):
        for i in range(0, len(training_set_positive)):
            if word_list[word] in training_set_positive[i]:
                count += 1
            dict_of_positive_reviews.update({word_list[word]: count})
        count = 0

    # I do everything again, only this time for the negative dataset
    # replace everything that is not alphanumeric or a whitespace using regex. I repalce with a whitespace here becuase it joined words together when i used ''
    training_set_negative = [re.sub("'", '', i) for i in
                             training_set_negative]
    training_set_negative = [re.sub('[^a-zA-Z ]', ' ', i) for i in
                             training_set_negative]
    training_set_negative = [i.lower() for i in training_set_negative]  # Convert everything to lowercase

    # This block of of code is for getting the total number of words in the positive training dataset after they are split
    training_data_test = [i.split() for i in training_set_negative]  # Split everything into separate words
    training_data_no_sublist = []  # This gets rid of sublists and creates a list where words are longer than specified length
    for lists in training_data_test:
        for word in lists:
            training_data_no_sublist.append(word)
    df_test = pd.DataFrame(training_data_no_sublist)
    # This value is then used in Task 5
    total_words_in_negative_reviews = len(df_test)

    # Here I search through the negative dataset, count each occurrence of the words specified in the word list
    # I add the final count of each word occurrence and add it to the dictionary with the word value
    count = 0
    for word in range(0, len(word_list)):
        for i in range(0, len(training_set_negative)):
            if word_list[word] in training_set_negative[i]:
                count += 1
            dict_of_negative_reviews.update({word_list[word]: count})
        count = 0



    return dict_of_positive_reviews, dict_of_negative_reviews, total_words_in_positive_reviews, total_words_in_negative_reviews


def Task4(positive_dictionary, negative_dictionary, tot_pos_words, tot_neg_words):


    values_list_positive = list(positive_dictionary.items())
    values_list_negative = list(negative_dictionary.items())

    alpha = 1

    dict_likelihood_positive, dict_likelihood_negative = {}, {}

    # Here I calculate the likelihood of each word in the positive and negative dictionaries
    for i in range(0, len(positive_dictionary)):
        likelihood_pos = ((values_list_positive[i][1]) + alpha) / (tot_pos_words + (alpha * len(positive_dictionary)))
        dict_likelihood_positive.update(({values_list_positive[i][0]: likelihood_pos}))

    for i in range(0, len(negative_dictionary)):
        likelihood_neg = ((values_list_negative[i][1]) + alpha) / (tot_neg_words + (alpha * len(negative_dictionary)))
        dict_likelihood_negative.update(({values_list_negative[i][0]: likelihood_neg}))

    # Here I calculate the posterior probability of each word in the positive and negative dictionaries
    prior_positive = tot_pos_words / (tot_pos_words + tot_neg_words)
    prior_negative = tot_neg_words / (tot_pos_words + tot_neg_words)


    return dict_likelihood_positive, dict_likelihood_negative, prior_positive, prior_negative


def Task5(review_text, dict_likelihood_positive, dict_likelihood_negative, prior_positive, prior_negative):

    # replace everything that is not alphanumeric or a whitespace using regex. I repalce with a whitespace here because it joined words together when i used ''
    review_text = re.sub('[^a-zA-Z ]', ' ',
                         review_text)
    review_text = review_text.lower()  # Convert everything to lowercase
    review_text = review_text.split()

    numerator = 0  # This is the numerator of the posterior probability
    denominator = 0  # This is the denominator of the posterior probability

    # Here I calculate the numerator and denominator of the posterior probability
    for i in range(0, len(review_text)):
        if review_text[i] in dict_likelihood_positive:
            numerator = numerator + (math.log(dict_likelihood_positive[review_text[i]]))
        if review_text[i] in dict_likelihood_negative:
            denominator = denominator + math.log(dict_likelihood_negative[review_text[i]])



    # Here I decide on the return value based on which number is greater
    if (numerator - denominator) > (
            math.log(prior_negative) - math.log(prior_positive)):  # used to avoid divisible by zero error
        return "positive"
    else:
        return "negative"


def Task6(training_data, label_data, test_data1, test_labels1):


    dataframe = pd.DataFrame({"Training_Data": training_data, "Training_Labels": label_data})
    dataframe = dataframe.reset_index(drop=True)

    folds = 10

    kf = KFold(n_splits=folds, shuffle=False)
    kf.get_n_splits(dataframe)


    training_mean_accuracy_score = 0    # accuracy score (average) for training data
    optimal_word_length = 0
    training_best_score = 0             # val that keeps updating with the better score for different word lengths

    for word_length in range(10, 0, -1):

        for train_index, test_index in kf.split(dataframe):
            # 4 dataframes/Series for training and testing.
            X_train, X_test = dataframe.loc[train_index, "Training_Data"], dataframe.loc[test_index, "Training_Data"]
            y_train, y_test = dataframe.loc[train_index, "Training_Labels"], dataframe.loc[test_index, "Training_Labels"]


            df = pd.DataFrame({"Training_Data": X_train, "Training_Labels": y_train})

            # Here I get the word list by calling Task2
            word_list = Task2(df['Training_Data'], word_length, 300)

            # Here I get the positive and negative dictionaries as well as the total number of positive and negative words
            pos_dict, neg_dict, total_positive_words, total_negative_words = Task3(df['Training_Data'],
                                                                                   df['Training_Labels'],
                                                                                   word_list)

            # Here I get the likelihood (dictionaries) and the prior probabilities
            dict_likelihood_positive, dict_likelihood_negative, prior_positive, prior_negative = Task4(pos_dict, neg_dict,
                                                                                                       total_positive_words,
                                                                                                       total_negative_words)
            # This value is used to keep track of if the prediction is correct or not (when calling task 5)
            correct_predictions = 0
            # The above value is then used in conjunction with this value to get the accuracy score
            total_predictions = len(X_test)

            X_test = X_test.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            for i in range(0, len(X_test)):

                answer = Task5(X_test[i], dict_likelihood_positive, dict_likelihood_negative, prior_positive,
                               prior_negative)
                # Here I check if the prediction is correct or not and update the value of correct_predictions
                if answer == y_test[i]:
                    correct_predictions += 1

            # Here I calculate the accuracy score for the current word length
            classification_accuracy = correct_predictions / total_predictions

            # Here I update the training_mean_accuracy_score
            training_mean_accuracy_score = training_mean_accuracy_score + classification_accuracy

        # Here I calculate the average accuracy score for the current word length
        training_mean_accuracy_score = training_mean_accuracy_score / folds
        print("Word Length", word_length, "Mean Score", training_mean_accuracy_score)

        # If the current score is better than the previous best score, I update the best score and the optimal word length
        if training_mean_accuracy_score > training_best_score:
            training_best_score = training_mean_accuracy_score
            optimal_word_length = word_length

        # Here I reset the training_mean_accuracy_score for the next word length
        training_mean_accuracy_score = 0

    print("\nOptimal word length:", optimal_word_length, ". Training best score", training_best_score)


    # Now I do the same thing for the test data

    X_test = test_data1.reset_index(drop=True)
    y_test = test_labels1.reset_index(drop=True)

    word_list = Task2(X_test, optimal_word_length, 300)

    pos_dict, neg_dict, total_positive_words, total_negative_words = Task3(X_test,
                                                                           y_test,
                                                                           word_list)

    dict_likelihood_positive, dict_likelihood_negative, prior_positive, prior_negative = Task4(pos_dict, neg_dict,
                                                                                               total_positive_words,
                                                                                               total_negative_words)

    correct_predictions = 0
    total_predictions = len(X_test)
    y_pred = []






    # Here I get the predictions for the test data
    for i in range(0, len(X_test)):

        answer = Task5(X_test[i], dict_likelihood_positive, dict_likelihood_negative, prior_positive,
                       prior_negative)
        if answer == y_test[i]:
            correct_predictions += 1
    # Here I add the prediction to the list of predictions. this will be used for the confusion matrix
        y_pred.append(answer)

    # Here I calculate the accuracy score for the optimal word length
    model_accuracy_test_set = correct_predictions / total_predictions

    print("\nModel Accuracy:", model_accuracy_test_set)

    # Here I create the y_true for the confusion matrix. I reset the index of the test_labels1
    y_true = test_labels1.reset_index(drop=True)


    # Here I encode the y_true and y_pred, with 0/1 for the confusion matrix
    for i in range(0, len(y_true)):
        if y_true[i] == 'positive':
            y_true[i] = 1
        elif y_true[i] == 'negative':
            y_true[i] = 0

    for i in range(0, len(y_pred)):
        if y_pred[i] == 'positive':
            y_pred[i] = 1
        elif y_pred[i] == 'negative':
            y_pred[i] = 0

    # Convert the series to a list
    y_true = y_true.tolist()

    # Here I create the confusion matrix
    confusion = metrics.confusion_matrix(y_true, y_pred)
    print("Confusion", confusion)

    # Here I calculate the accuracy score for the confusion matrix
    accuracy = metrics.accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion.ravel()

    true_positive_rate = tp / (tp + fn)
    false_positive_rate = fp / (tn + fp)
    false_negative_rate = fn / (tp + fn)
    true_negative_rate = tn / (tn + fn)

    print("True Positive Rate", true_positive_rate, "\nFalse Positive Rate", false_positive_rate, "\nTrue Negative Rate",
          true_negative_rate, "\nFalse Negative Rate", false_negative_rate)

    print("Accuracy", accuracy)

    # Note: I have used 2 ways to calculate the accuracy score. This is because my first way was implemented and
    # then I realised it would not work for making the confusion matrix as it gave me issues. I then added the second
    # method. Both accuracy scores are printed to the console.


def main():
    training_data1, training_labels, test_data, test_labels= Task1()
    Task6(training_data1, training_labels, test_data, test_labels)



main()