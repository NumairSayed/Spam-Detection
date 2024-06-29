import streamlit as st
import re
import pandas as pd
import numpy as np
#from tqdm import tqdm

def process_message(message):
    words = re.split(r'[.,:;!?()\[\]"\s]+', message.lower())
    return [word for word in words if word]

def process_emails(file_path, encoding='ISO-8859-1'):
    data = []
    with open(file_path, 'r', encoding=encoding) as file:
        for line in file:
            line = line.strip()
            if line.startswith("ham") or line.startswith("spam"):
                label, message = line.split(',', 1)
                processed_message = process_message(message)
                data.append([label, processed_message])
    return data

def calculate_log_probability(label, dataframe, binary_wordmap):
    label_count = len(dataframe[dataframe['label'] == label])
    total_count = len(dataframe)
    log_prob = np.log(label_count + 1) - np.log(total_count + 2)  # Prior probability with Laplace smoothing

    for word, present in binary_wordmap.items():
        word_count = sum((word in words) for words in dataframe[dataframe['label'] == label]['words'])
        if present:
            log_prob += np.log(word_count + 1) - np.log(label_count + 2)
        else:
            log_prob += np.log(label_count - word_count + 1) - np.log(label_count + 2)
    
    return log_prob

def main():
    # Load the training data
    file_path = 'spam.txt'
    data = process_emails(file_path)
    training_dataframe = pd.DataFrame(data, columns=['label', 'words'])
    
    # Select 1000 random rows for testing
    #random_indices = np.random.choice(training_dataframe.index, size=1000, replace=False)
    #testing_dataframe = training_dataframe.loc[random_indices]

    # Remove these rows from the original dataframe
    #training_dataframe.drop(random_indices, inplace=True)
    #training_dataframe.reset_index(drop=True, inplace=True)
    #testing_dataframe.reset_index(drop=True, inplace=True)

    spam_stars = []
    for index, rows in training_dataframe.iterrows():
        if rows['label'] == 'spam':
            for word in rows['words']:
                spam_stars.append(word)
    spamstars = set(spam_stars)
    print(spamstars)
    # Create Streamlit UI
    st.title("Email Spam Detector")
    st.write("Paste your email content below to determine if it's spam or not.")
    
    st.markdown("""
    ### Numair Sayed
    I have tried explaining the concepts that I have used, feel free to reach out in case of suggestions and critique.
    ### Email-Address: sayednumair2019@gmail.com 
    #### Dataset used for classification: https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification
    ## Understanding the Naive Bayes Classifier
    The Naive Bayes classifier is a probabilistic classifier based on Bayes' Theorem with strong (naive) independence assumptions. 
    It assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

    ### Bayes' Theorem
    Bayes' Theorem describes the probability of an event, based on prior knowledge of conditions that might be related to the event.
    
    The formula for Bayes' Theorem is:
    $$
    P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}
    $$
    
    ### Maximum A Posteriori (MAP) Estimation
    The MAP estimate of a parameter is the mode of the posterior distribution, which incorporates both the prior distribution and the likelihood of the data.
    
    ### Laplace Smoothing
    Laplace Smoothing is used to handle the problem of zero probabilities in Naive Bayes. It adds a small fixed value (usually 1) to each count to ensure that no probability is ever zero.

    Consider we are calculating the probability of a word appearing in a spam email:
    $$
    P(word_i|spam) = \\frac{(count(word_i \; in \; spam) + 1)}{(count(spam) + |Vocabulary|)}
    $$

    ### Example: Review Confidence
    Let's say we want to find the confidence in good reviews for two merchants:
    - Merchant A: 9 out of 10 reviews are good.
    - Merchant B: 90 out of 100 reviews are good.

    Without Laplace Smoothing:
    - Merchant A: \( P(good|A) = 9/10 = 0.9 )
    - Merchant B: \( P(good|B) = 90/100 = 0.9 )

    With Laplace Smoothing:
    - Merchant A: \( P(good|A) = (9+1)/(10+2) = 10/12 = 0.833 )
    - Merchant B: \( P(good|B) = (90+1)/(100+2) = 91/102 = 0.892 )

    The intuitive explanation to this would be, adding 1 to numerator extends the event of you having a good experience and adding 2 to the 
    denominator extends the sample space to consider 2 more probable events when you enter into the sample space, i.e you may induce a good 
    and a bad review probabilistically. This helps heavily in dealing with datasets in which certain events have zero probability or
    calculating results for a data which is not yet seen but may appear in a question prompt.
    
    This suggest you should have better confidence in choosing the second merchant and that he has a better chance in your satisfaction as a customer.

    ### Laplace's Rule of Succession
    Laplace's Rule of Succession is used to estimate the probability of an event that has never occurred in the past.
    It states that if an event has occurred \( s \) times in \( n \) trials, the probability of the event occurring in the next trial is:
    $$
    P = \\frac{s + 1}{n + 2}
    $$
    This rule helps in handling the zero probability problem by adding 1 to the number of successes and 2 to the total number of trials.
    """)

    new_email_message = st.text_area("Enter the email message:")
    
    if st.button("Check Email"):
        input_list = process_message(new_email_message)
        binary_wordmap = {word: (word in input_list) for word in spam_stars}

        spam_in_training = len(training_dataframe[training_dataframe['label'] == 'spam'])
        ham_in_training = len(training_dataframe) - spam_in_training
        total_spam = spam_in_training
        total_ham = ham_in_training

        param_sum_of_spam = calculate_log_probability('spam', training_dataframe, binary_wordmap) + np.log(total_spam/len(training_dataframe))
        param_sum_of_ham  = calculate_log_probability('ham', training_dataframe, binary_wordmap) + np.log(total_ham/len(training_dataframe))

        st.write("### Log Probability Calculation")
        st.write(f"Log probability of being spam: {param_sum_of_spam}")
        st.write(f"Log probability of being ham: {param_sum_of_ham}")

        if param_sum_of_spam > param_sum_of_ham:
            st.write("### Result: The email is classified as **SPAM**.")
        else:
            st.write("### Result: The email is classified as **HAM**.")

if __name__ == '__main__':
    main()

