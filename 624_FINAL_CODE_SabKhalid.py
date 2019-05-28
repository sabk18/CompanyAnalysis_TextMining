#Final Project for EM 624:-->

import pandas as pd
import numpy as np



data =pd.read_csv("employee_reviews.csv")    #Load dataset
data.head()                            #prints the first top 5

#emoving columns that are NOT needed for this analysis:
data.drop(['location','dates','job-title','summary','advice-to-mgmt','helpful-count','link'],axis=1,inplace = True)
#print data
#data.info()

#-----First organize/clean the data within the dataframe-------

#Change "none" values in all the ratinf columns to "0":
data['work-balance-stars']=np.where(data['work-balance-stars']=='none', 0, data['work-balance-stars'])
data['culture-values-stars']=np.where(data['culture-values-stars']=='none', 0, data['culture-values-stars'])
data['carrer-opportunities-stars']=np.where(data['carrer-opportunities-stars']=='none', 0, data['carrer-opportunities-stars'])
data['comp-benefit-stars']=np.where(data['comp-benefit-stars']=='none', 0, data['comp-benefit-stars'])
data['senior-mangemnet-stars']=np.where(data['senior-mangemnet-stars']=='none', 0, data['senior-mangemnet-stars'])

#print data
#rename the dataframe columns for the ratings forc onvenience:
data.rename(columns={'work-balance-stars':'work-stars','culture-values-stars':'culture-stars','carrer-opportunities-stars':'career-stars','comp-benefit-stars':'benefit-stars','senior-mangemnet-stars':'management-stars'},inplace = True)
#print data

#change the ratings(object) to float:
data['overall-ratings']=data['overall-ratings'].astype(dtype=np.float64)
data['work-stars']=data['work-stars'].astype(dtype=np.float64)
data['culture-stars']=data['culture-stars'].astype(dtype=np.float64)
data['career-stars']=data['career-stars'].astype(dtype=np.float64)
data['benefit-stars']=data['benefit-stars'].astype(dtype=np.float64)
data['management-stars']=data['management-stars'].astype(dtype=np.float64)

#print data.shape()
print data.describe()
print data.head(5)
print data

#"PLOT the ratings by company"
import matplotlib.pyplot as plt
import seaborn as sns

# = data[data['company']=='google'][:]
#print google
#pl.figure(figsize =(10,5))

sns.set(style="darkgrid")
sns.countplot(x="overall-ratings", hue="company",data=data)
plt.xlabel("Overall-Ratings",fontsize=15)
plt.figure()
plt.show()

sns.countplot(x="work-stars", hue="company",data=data)
plt.xlabel("Work Stars",fontsize=15)
plt.figure()
#plt.show()

sns.countplot(x="culture-stars", hue="company",data=data)
plt.xlabel("Culture Stars",fontsize=15)
plt.figure()
#plt.show()

sns.countplot(x="career-stars", hue="company",data=data)
plt.xlabel("Career Stars",fontsize=15)
plt.figure()
#plt.show()

sns.countplot(x="benefit-stars", hue="company",data=data)
plt.xlabel("Benefit Stars",fontsize=15)
plt.figure()
#plt.show()

sns.countplot(x="management-stars", hue="company",data=data)
plt.xlabel("Management Stars",fontsize=15)
plt.figure()

plt.show()

#Now we will extract the pros and cons for each company forming a new database for each


google_pros = data[data['company']=='google']['pros']
google_cons = data[data['company']=='google']['cons']

amazon_pros = data[data['company']=='amazon']['pros']
amazon_cons = data[data['company']=='amazon']['cons']

facebook_pros = data[data['company']=='facebook']['pros']
facebook_cons = data[data['company']=='facebook']['cons']

netflix_pros = data[data['company']=='google']['pros']
netflix_cons = data[data['company']=='google']['cons']

apple_pros = data[data['company']=='google']['pros']
apple_cons = data[data['company']=='google']['cons'] 

microsoft_pros = data[data['company']=='microsoft']['pros']
microsoft_cons = data[data['company']=='microsoft']['cons']      
    
#now we will conver the columns of pros and cons for each company into lists:   
def review_lists(pros,cons):    
    pro_list=pros.values.tolist()
    con_list=cons.values.tolist()
    return pro_list, con_list
    
google_pro_list,google_con_list = review_lists(google_pros,google_cons)
amazon_pro_list,amazon_con_list = review_lists(amazon_pros,amazon_cons)
facebook_pro_list,facebook_con_list = review_lists(facebook_pros,facebook_cons)
netflix_pro_list,netflix_con_list = review_lists(netflix_pros,netflix_cons)
apple_pro_list,apple_con_list = review_lists(apple_pros,apple_cons)
microsoft_pro_list,microsoft_con_list = review_lists(microsoft_pros,microsoft_cons)


#create a function to Clean our above lists:

def clean_lists(list):
    import re
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    reviews =[]

    for review in list:
        #print review
        review = re.sub(r"http\S+", "",review) #removes URL (if any)
        #print review
        review =re.sub(r"^\S+", "",review)     #removes extra spaces from start and end
        #print review
        tokens = review.split()                 ##split into words:
        #print tokens
        tokens =[w.lower() for w in tokens if w.isalpha()] #convert to a lower case and removes words that are not alphabetic including puntuation:
        #print tokens
        stop_words =set(stopwords.words("english"))          #removing stopwords:
        tokens = [w for w in tokens if w not in stop_words]
        #print tokens
        detokenized_tweets = TreebankWordDetokenizer().detokenize(tokens) #Detokenize back into string
        #print detokenized_tweets
        reviews.append(detokenized_tweets)
        #print reviews[:20]

    cleaned_tweets = filter(None,reviews)  #removes empty strings within the list    
    #print cleaned_tweets
    #print len(cleaned_tweets) 
    return cleaned_tweets
    
google_clean_pro_list= clean_lists(google_pro_list)
google_clean_con_list= clean_lists(google_con_list)

amazon_clean_pro_list= clean_lists(amazon_pro_list)
amazon_clean_con_list= clean_lists(amazon_con_list)

facebook_clean_pro_list= clean_lists(facebook_pro_list)
facebook_clean_con_list= clean_lists(facebook_con_list)

netflix_clean_pro_list= clean_lists(netflix_pro_list)
netflix_clean_con_list= clean_lists(netflix_con_list)

apple_clean_pro_list= clean_lists(apple_pro_list)
apple_clean_con_list= clean_lists(apple_con_list) 
   
microsoft_clean_pro_list= clean_lists(microsoft_pro_list)
microsoft_clean_con_list= clean_lists(microsoft_con_list)    
  
#extract bigrams:
def bigram_chk(list_):
    from collections import Counter
    import nltk

    token=' '.join(list_).split()
    
    word_count = Counter(token).most_common(15) 
    bigrams=list(nltk.bigrams(token))
    #print bigrams
    #return len(bigrams)

    #Extracting the top 10 most frequent bigrams from our cleaned list:
    bigrams_count = Counter(bigrams).most_common(15)
    
    return bigrams_count, word_count  
    
google_pro_bigram, google_pro_top_10_words=bigram_chk(google_clean_pro_list)
print "\nThe top 15 bigrams for the posisitve comments at google are: \n" +str(google_pro_bigram)
google_con_bigram,google_con_top_10_words =bigram_chk(google_clean_con_list)
print "\nThe top 15 bigrams for the negative comments at google are: \n" +str(google_con_bigram)
  
amazon_pro_bigram,amazon_pro_top_10_words =bigram_chk(amazon_clean_pro_list)
print "\nThe top 15 bigrams for the posisitve comments at amazon are: \n" +str(amazon_pro_bigram)
amazon_con_bigram,amazon_con_top_10_words =bigram_chk(amazon_clean_con_list)
print "\nThe top 15 bigrams for the negative comments at amazon are: \n" +str(amazon_con_bigram)


facebook_pro_bigram,facebook_pro_top_10_words =bigram_chk(facebook_clean_pro_list)
print "\nThe top 15 bigrams for the posisitve comments at facebook are: \n" +str(facebook_pro_bigram)
facebook_con_bigram, facebook_con_top_10_words =bigram_chk(facebook_clean_con_list)
print "\nThe top 15 bigrams for the negative comments at facebook are: \n" +str(facebook_con_bigram)

netflix_pro_bigram,netflix_pro_top_10_words =bigram_chk(netflix_clean_pro_list)
print "\nThe top 15 bigrams for the posisitve comments at netflix are: \n" +str(netflix_pro_bigram)
netflix_con_bigram,netflix_con_top_10_words =bigram_chk(netflix_clean_con_list)
print "\nThe top 15 bigrams for the negative comments at netflix are: \n" +str(netflix_con_bigram)

microsoft_pro_bigram,microsoft_pro_top_10_words =bigram_chk(microsoft_clean_pro_list)
print "\nThe top 15 bigrams for the posisitve comments at microsoft are: \n" +str(microsoft_pro_bigram)
microsoft_con_bigram, microsoft_con_top_10_words =bigram_chk(microsoft_clean_con_list)
print "\nThe top 15 bigrams for the negative comments at microsoft are: \n" +str(microsoft_con_bigram)  

#Now print top 10 words for each pos and neg review for each company:

print "\nThe top 10 words in the positive comments at google are: \n" +str(google_pro_top_10_words)
print "\nThe top 10 words in the negative comments at google are: \n" +str(google_con_top_10_words)

print "\nThe top 10 words in the positive comments at amazon are: \n" +str(amazon_pro_top_10_words)
print "\nThe top 10 words in the positive comments at amazon are: \n" +str(amazon_con_top_10_words)

print "\nThe top 10 words in the positive comments at facebook are: \n" +str(facebook_pro_top_10_words)
print "\nThe top 10 words in the negative comments at facebook are: \n" +str(facebook_con_top_10_words)

print "\nThe top 10 words in the positive comments at netflix are: \n" +str(netflix_pro_top_10_words)
print "\nThe top 10 words in the negative comments at netflix are: \n" +str(netflix_con_top_10_words)

print "\nThe top 10 words in the positive comments at microsoft are: \n" +str(microsoft_pro_top_10_words)
print "\nThe top 10 words in the negative comments at microsoft are: \n" +str(microsoft_con_top_10_words)


#HISTOGRAM for POS:
a = ('Google Pro Reviews')  
b =('Google Con Reviews')
c = ('Amazon Pro Reviews')  
d =('Amazon Con Reviews')
e = ('Facebook Pro Reviews')  
f =('Facebook Con Reviews')
g = ('Netflix Pro Reviews')  
h =('Netlfix Con Reviews')  
i = ('Microsoft Pro Reviews')  
j =('Microsoft Con Reviews')

def plot_words(top_words, title):
    count_df=pd.DataFrame(top_words,columns=['word','frequency'])
    count_df.plot(kind='bar',x='word',y='frequency',color='grey')
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()   
    
googlePro_words =plot_words(google_pro_top_10_words,a )
googleCon_words =plot_words(google_con_top_10_words,b)

amazonPro_words =plot_words(amazon_pro_top_10_words,c)
amazonCon_words =plot_words(amazon_con_top_10_words,d)

facebookPro_words =plot_words(facebook_pro_top_10_words,e)
facebookCon_words =plot_words(facebook_con_top_10_words,f)

netflixPro_words =plot_words(netflix_pro_top_10_words,g)
netflixCon_words =plot_words(netflix_con_top_10_words,h)

microsoftPro_words =plot_words(microsoft_pro_top_10_words,i)
microsoftCon_words =plot_words(microsoft_con_top_10_words,j)

                            
                      
a = ('Google Pro Reviews')  
b =('Google Con Reviews')
c = ('Amazon Pro Reviews')  
d =('Amazon Con Reviews')
e = ('Facebook Pro Reviews')  
f =('Facebook Con Reviews')
g = ('Netflix Pro Reviews')  
h =('Netlfix Con Reviews')  
i = ('Microsoft Pro Reviews')  
j =('Microsoft Con Reviews')
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#Creating a word cloud for pros and cons lists:

def word_clouds(list_, title):
    
    #Transforming the list into a string
    str_ = ' '.join(list_)

    stopwords =set(STOPWORDS)
    #removing any extra bigger words from these wordclouds
    stop_words =["want","company","lot","many","work"]
    new_stopwords = stopwords.union(stop_words)

#Defining the wordcloud parameters and #Generate word cloud
    wc = WordCloud(background_color="white", max_words=200,max_font_size=40,scale=3,random_state = 42, stopwords=new_stopwords).generate(str_)
    plt.figure(figsize=(30,30))

#store to file
    wc.to_file('company.png')

#Show the cloud
    plt.imshow(wc)
    plt.axis('off')
    plt.title(title)
    plt.show() 
    return
    
word_clouds(google_clean_pro_list, a)
word_clouds(google_clean_con_list, b)

word_clouds(amazon_clean_pro_list, c)
word_clouds(amazon_clean_con_list, d)

word_clouds(facebook_clean_pro_list, e)
word_clouds(facebook_clean_con_list, f)

word_clouds(netflix_clean_pro_list, g)
word_clouds(netflix_clean_con_list, h)

word_clouds(microsoft_clean_pro_list, i)
word_clouds(microsoft_clean_con_list, j) 

#print "\n List after cleaning is: \n" +str( google_clean_pro_list)  

###----Sentimental Analysis-----###


#def sentiment(list_):
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

 
def sentiment(list_):

    from textblob import TextBlob
    sentence =' '.join(list_)
    review_sentiment =TextBlob(sentence).sentiment
    return review_sentiment
    
gp_review_sentiment=sentiment(google_clean_pro_list)
gn_review_sentiment=sentiment(google_clean_con_list)
ap_review_sentiment=sentiment(amazon_clean_pro_list)
an_review_sentiment=sentiment(amazon_clean_con_list)
fp_review_sentiment=sentiment(facebook_clean_pro_list)
fn_review_sentiment=sentiment(facebook_clean_con_list)
np_review_sentiment=sentiment(netflix_clean_pro_list)
nn_review_sentiment=sentiment(netflix_clean_con_list)
mp_review_sentiment=sentiment(microsoft_clean_pro_list)
mn_review_sentiment=sentiment(microsoft_clean_con_list)

print "\nSentiment for pro reviews for Google : \n" + str(gp_review_sentiment)
print "\nSentiment for con reviews for Google : \n" + str(gn_review_sentiment)

print "\nSentiment for pro reviews for amazon : \n" + str(ap_review_sentiment)
print "\nSentiment for con reviews for amazon : \n" + str(an_review_sentiment)

print "\nSentiment for pro reviews for Facebook : \n" + str(fp_review_sentiment)
print "\nSentiment for con reviews for Facebook : \n" + str(fn_review_sentiment)

print "\nSentiment for pro reviews for Netflix : \n" + str(np_review_sentiment)
print "\nSentiment for con reviews for Netflix : \n" + str(nn_review_sentiment)

print "\nSentiment for pro reviews for Microsoft : \n" + str(mp_review_sentiment)
print "\nSentiment for con reviews for Microsoft : \n" + str(mn_review_sentiment)





