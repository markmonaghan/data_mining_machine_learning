import pandas as pd
import re
import csv

positive_list = []
with open('positive-words.txt', 'r') as file:
    positive_list = file.read().splitlines()
postive_keys = set(positive_list)

negative_list = []
with open('negative-words.txt', 'r') as file:
    negative_list = file.read().splitlines()
negative_keys = set(negative_list)

count_pos = 0
count_neg = 0
count_both = 0
games = pd.read_csv('/Users/rohit/MSc AI/group-coursework-ha/data/raw/games.csv')
cleaned_reviews = []
reviews = games['Reviews']
review_flag = 0
confused_dataset = []

for review in reviews:
    texts = review[1:]
    texts = texts[:-1]
    if(len(texts) == 0):
        break
    line_present = True
    while(line_present):
        try:
            pos = False
            neg = False
            current_review = (texts[1:texts.find(texts[0], texts.find(texts) + 1)]).strip() # removing the quotes
            original_texts = current_review
            if(len(current_review) > 0):
                current_review = re.sub(r'\W', ' ', str(current_review)) # removing single character
                current_review = re.sub(r'\s+[a-zA-Z]\s+', ' ', current_review) # removing single words
                current_review = re.sub(r'\^[a-zA-Z]\s+', ' ', current_review) # removing special characters
                if(len(current_review.strip()) == 0):
                    # if the current sentence length is zero skip and move to next sentence if present
                    remaining_texts = texts.split(original_texts, 1)
                    if (len(remaining_texts) > 1):
                        texts = remaining_texts[1][2:]
                        texts = texts.strip()
                        if (len(texts) == 0):
                            line_present = False
                    continue
                current_review = current_review.replace("\\n", ' ') # removing new lines
                current_review = re.sub(r"\s+", " ", current_review) # removing multiple spaces
                current_review = current_review.lower()
                pos = any(ele in current_review for ele in positive_list)
                neg = any(negele in current_review for negele in negative_list)
                if((pos == False and neg == False)):
                    # if the sentence doesn't have positive or negative words then skip and go to next sentence if present
                    remaining_texts = texts.split(original_texts, 1)
                    if (len(remaining_texts) > 1):
                        texts = remaining_texts[1][2:]
                        texts = texts.strip()
                        if (len(texts) == 0):
                            line_present = False
                    continue
                if(pos == True and neg == False):
                    review_flag = 1
                    count_pos += 1
                if (pos == False and neg == True):
                    review_flag = 0
                    count_neg += 1
                if (pos == True and neg == True):
                    # Some sentences had both positive and negative values
                    # Hence finding out the max occurence of postive and negative and assigning based on that
                    # Manually verified by printing. Ignoring the zeros
                    pos_values = [current_review.count(key) for key in postive_keys]
                    neg_values = [current_review.count(key) for key in negative_keys]
                    avg = sum(pos_values) - sum(neg_values)
                    if(avg > 0):
                        review_flag = 1
                        count_pos += 1
                    if(avg < 0):
                        review_flag = 0
                        count_neg += 1
                    if(avg == 0):
                        confused_dataset.append([current_review, -1])
                        # if the avg is zero skip and move to the next senstence if present
                        remaining_texts = texts.split(original_texts, 1)
                        if (len(remaining_texts) > 1):
                            texts = remaining_texts[1][2:]
                            texts = texts.strip()
                            if (len(texts) == 0):
                                line_present = False
                        continue
                cleaned_reviews.append([current_review, review_flag])
                remaining_texts = texts.split(original_texts, 1)
                if(len(remaining_texts) > 1):
                    texts = remaining_texts[1][2:]
                    texts = texts.strip()
                    if(len(texts) == 0):
                        line_present = False
                else:
                    line_present = False
            else:
                line_present = False
        except:
            print(texts)
print(len(cleaned_reviews))
print(count_pos)
print(count_neg)
print(count_both)
# with open('data_cleaning_games.csv', 'w', encoding='UTF-8', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(cleaned_reviews)

with open('confused_data.csv', 'w', encoding='UTF-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(confused_dataset)
