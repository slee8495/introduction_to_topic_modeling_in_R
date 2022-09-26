# Topic Mdoeling

# Task 1: Loading the packages, learning about LDA, and loading the data 

## Latent Dirichlet allocation

# Uunsupervised learning (including topic modeling) is used when we do not know the categories that we are interested in uncovering. 
# In topic modeling, documents are not assumed to belong to one topic or category, but simultaneously can belong to several topics. The topic distributions also vary over documents. 
# 
# The workhorse function for the topic model is `LDA`, which stands for Latent Dirichlet Allocation, the technical name for this particular kind of model. 
# 
# We will now use a dataset that contains the lead paragraph of around 5,000 articles about the economy published in the New York Times between 1980 and 2014. 
# 
# Preprocess the text using the standard set of techniques.
# 
# The number of topics in a topic model is somewhat arbitrary, and set by you the researcher, 
# so you need to play with the number of topics to see if you get anything more meaningful. 
# We start here with 30 topics.


# First, let's load the packages
library(quanteda)
library(quanteda.textplots)              
library(topicmodels) 


## Let's look at loading and cleaning the data

# around 5,000 articles about the economy published in the New York Times between 1980 and 2014.

# reading data and preparing corpus object

nyt <- read.csv("nytimes.csv", stringsAsFactors = FALSE)

as.character(nyt$lead_paragraph[1])

# Now cleaning the data 


cdfm <- quanteda::tokens(nyt$lead_paragraph, 
                         remove_punct = TRUE,
                         remove_symbols = TRUE,
                         remove_numbers = TRUE,
                         remove_url = TRUE,
                         remove_separators = TRUE,
                         split_hyphens = TRUE,
                         include_docvars = TRUE)

cdfm <- quanteda::tokens_select(cdfm, pattern = stopwords("en"), selection = "remove")


cdfm <- quanteda::dfm(cdfm, #Document Feature Matrix 
                      verbose = quanteda_options("verbose"),
                      tolower = TRUE) 




cdfm <- quanteda::dfm_trim(cdfm, 
                           min_termfreq = 50, 
                           min_docfreq = 5)
?dfm_trim

#min_termfreq minimum values of feature frequencies across all documents, below/above which features will be removed
#min_docfreq  minimum values of a feature's document frequency, below/above which features will be removed

#Task 2

## Second, let's look at LDA -- What is it?


#Task 2 Estimate LDA with K Topics

# Trade off between topic specificity and cohesion. Note that the machine will give you how ever many number of topics that you ask! 

set.seed(12345)
K <- 100
lda <- LDA(cdfm, k = K, method = "Gibbs", 
           control = list(verbose=25L, seed = 123, burnin = 100, iter = 500))

#Try running it with a smaller number of topics

set.seed(12345)
K <- 30
lda <- LDA(cdfm, k = K, method = "Gibbs", 
          control = list(verbose=25L, seed = 123, burnin = 100, iter = 500))


#Task 3 Extract Terms 

#We can use `get_terms` to the top `n` terms from the topic model, and `get_topics` to predict the top `k` topic for each document. This will help us interpret the results of the model.

terms <- get_terms(lda, 15)
as.data.frame(terms)



terms <- get_terms(lda, 15)
topics <- get_topics(lda, 1)
head(topics, 10)


#Let's take a closer look at some of these topics. To help us interpret the output, we can look at the words associated with each topic and take a random sample of documents highly associated with each topic.

# Topic 1

cat("Topic 1:",
    "\n",
    "Topic Words:", paste(terms[,1], collapse=", "), "\n", "Sampled Text: ",
     "\n",
    sample(nyt$lead_paragraph[topics==1], 1), sep = "")


# Topic 9 and 10 about the Central Bank

cat("Topic 9:",
    "\n",
    "Topic Words:", paste(terms[,9], collapse=", "), "\n", "Sampled Text: ",
    "\n",
    sample(nyt$lead_paragraph[topics==9], 1), sep = "")

cat("Topic 24:",
    "\n",
    "Topic Words:", paste(terms[,24], collapse=", "), "\n", "Sampled Text: ",
    "\n",
    sample(nyt$lead_paragraph[topics==24], 1), sep = "")


# You will that often some topics do not make much sense. They just capture the remaining cluster of words, and often correspond to stopwords. For example:

# Topic 8

cat("Topic 2:",
    "\n",
    "Topic Words:", paste(terms[,2], collapse=", "), "\n", "Sampled Text: ",
    "\n",
    sample(nyt$lead_paragraph[topics==2], 1), sep = "")




# In the case of date with timestamps, looking at the evolution of certain topics over time can also help interpret their meaning. 
# Let's look for example at Topic 3, which appears to be related to jobs and the labor market.

# Topic 3

cat("Topic 3:",
    "\n",
    "Topic Words:", paste(terms[,3], collapse=", "), "\n", "Sampled Text: ",
    "\n",
    sample(nyt$lead_paragraph[topics==3], 1), sep = "")



# add predicted topic to dataset


nyt$pred_topic <- topics
nyt$year <- substr(nyt$datetime, 1, 4) # extract year




# We can then make a frequency table with articles about the labor market, per year


tab <- table(nyt$year[nyt$pred_topic==3])
plot(tab, ylab = "Frequency of Labor Market Announcements", main = "Attention to the labor market \n in the NYT over time")


#But we can actually do better than this. LDA is a probabilistic model, which means that for each document, it actually computes a distribution over topics. In other words, each document is considered to be __about a mixture of topics__. 
#This information is included in the matrix `gamma` in the LDA object (`theta` in the notation we used for the slides). For example, article 1 is 7% about topic 3, 7% about topic 14, etc 

round(lda@gamma[1,], 2)


# So we can actually take the information in the matrix and aggregate it to compute the average probability that an article each year is about a particular topic. Let's now choose Topic 24, which appears to be related to the U.S. central bank

cat("Topic 24:",
    "\n",
    "Topic Words:", paste(terms[,24], collapse=", "), "\n", "Sampled Text: ",
     "\n",
    sample(nyt$lead_paragraph[topics==24], 1), sep = "")


# Add probability to original dataframe `nyt` and w aggregate  it at the year level


nyt$prob_topic <- lda@gamma[,24]

# now aggregate at the year level
agg <- aggregate(nyt$prob_topic, by=list(year=nyt$year), FUN=mean)


# and plot it

plot(agg$year, agg$x, type="l", xlab="Year", ylab="Avg. prob. of article about topic 24",
     main="Estimated proportion of articles about the US central bank")

