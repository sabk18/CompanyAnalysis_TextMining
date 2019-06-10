# TEXT-MINING-Company_Analysis

## Introduction

This project uses traditional NLP to analyze raw text, which in this dataset are employee reviews on the top six tech companies:

  * Google
  * Amazon
  * Facebook
  * Netflx
  * Apple
  * Microsoft
  
**(The reviews are taken from Glassdoor)**

## Goals

The main goal of this project is to study and analyze reviews and ratings from current and former employees for Google, Amazon, Facebook, Netflix and Microsoft. 

**Value** 
This analysis can help understand which company is a preferable choice to work for by understanding the sentiments and ratings of employees who have experienced working at these companies.

Project Goals are as following:
  *	Analysis of Employee overall rating
  *	Analysis of Work/Life Balance ratings
  *	Analysis of Culture Rating
  * Analysis of Career Opportunities Ratings
  * Analysis of Benefits Ratings
  * Analysis of Senior Management Ratings
  * Sentimental Analysis of Pros/cons for each company

## Data Preparation

The dataset was opened and read in Python. The dataset was manipulated and organized using Pandas. 
For each pos/con review within a list:
1.	Removed any URLs if any
2.	Removed extra spaces before and after the text
3.	Tokenized the reviews into “bag of words”
4.	Converted the words to lower case
5.	Removed punctuation and nonalphabetic values
6.	Removed stop words 
7.	Detokenized the words into sentences
8.	Collected these cleaned reviews back into a list
9.	Removed any empty strings from the list

## Data Representation

To visualize my analysis I used matplotlib and Seaborn to visualize my charts. 
Wordclouds were also generated for are cleaned bag of words and Bigramms.







