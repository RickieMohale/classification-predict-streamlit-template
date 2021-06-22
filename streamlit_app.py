"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import seaborn as sns
import numpy as np


## nlpk import 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import TreebankWordTokenizer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
## other imports
from PIL import Image
import re
import string

#suppress cell_warnings
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load data 
train_df = pd.read_csv("resources/train.csv")

# defining a data Processing and cleaning  function
def tweet_processor(input_tweet):


    # Removing https and trailing white spaces
	input_tweet = re.sub(r'^RT ','', re.sub(r'https://t.co/\w+', '', input_tweet).strip()) 

    # Removing puctuations
	input_tweet = input_tweet.lower()
	tweet = ''.join([x for x in input_tweet if x not in string.punctuation])

	# Tweet tokenizing
	tokeniser = TreebankWordTokenizer()
	tokens = tokeniser.tokenize(data)
	

    # Removing stopwords
	tokens_wt_stopwords = [word for word in tokens if word not in stopwords.words('english')]
   

    
	pos = pos_tag(tokens_wt_stopwords)


    # Lemmatization
	lemmatizer = WordNetLemmatizer()
	tweet = ' '.join([lemmatizer.lemmatize(word, po[0].lower()) if po[0].lower() in ['n', 'r', 'v', 'a'] else word for word, po in pos])
    # tweet = ' '.join([lemmatizer.lemmatize(word, 'v') for word in tweet])

	return tweet

# 
# Loading prediction model using its path
def load_model(path_):
	model_ = joblib.load(open(os.path.join(path_),"rb"))
	return model_


# Prediction classification 
def predict_class(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key
# The main function where we will build the actual app
def main():
	
	"""Tweet Classifier App with Streamlit """
    

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Tweet Classifer")


	pages = ["Prediction Page","Data Visualization","Project Team", "Company Information, Background & Team"]
	selection = st.sidebar.selectbox("Choose Page :", pages)

	# Building out the "Background" page
	if selection == "Company Information, Background & Team":
		st.info('Company Information and Background says who?')

		st.header('Our Mission')		
		st.write('To develop creative AI solutions for Africa\'s toughest problems.')

		st.header('Our Vision')
		st.write('A better and smarter Africa.')

		st.header('Our Amazing Team')
		st.write('A team of 6 passionate AI solutionists.')
		#First row of pictures

		col1, col2,col3 = st.beta_columns(3)
		Ric_Pic =Image.open('resources/imgs/Rickie_pic.png') 
		col1.image(Ric_Pic,caption="Rickie Mogale Mohale", width=150)
		col1.write('Tech-lead and software developer.')
        
		Cot_Pic =Image.open('resources/imgs/courtney_pic.png') 
		col2.image(Cot_Pic,caption="Courtney Murugan", width=150)
		col2.write('Machine learning engineer')

		Cot_Pic =Image.open('resources/imgs/jacques_pic.png') 
		col3.image(Cot_Pic,caption="Jacques Stander", width=150)
		col3.write('Project manager')

        #Second row of pictures
		col4, col5,col6 = st.beta_columns(3)
		vesh_Pic =Image.open('resources/imgs/veshen_pic.png') 
		col4.image(vesh_Pic,caption="Veshen Naidoo", width=150)
		col4.write('UQ Designer')
        
		Phiw_Pic =Image.open('resources/imgs/blue_pic.png') 
		col5.image(Phiw_Pic,caption="Phiweka Mthini", width=150)
		col5.write('Digital marketer ')

		nor_Pic =Image.open('resources/imgs/blue_pic.png') 
		col6.image(nor_Pic,caption="Nourhan ALfalous", width=150)
		col6.write('Data scientist')

		#Third row of picture 
		col7, col8,col9 = st.beta_columns(3)

		zin_Pic =Image.open('resources/imgs/zintle_pic.png') 
		col8.image(zin_Pic,caption='Zintle Faltein-Maqubela', width=150)
		col8.write("Supervisor")

		st.header('How we started?')
		st.write('African Intelligence started as a group of 6 students who met each other on a university project. The students bonded together around a love for solving problems with the help of AI. ')	
		st.write('These students all graduated with flying colours and entered succesful carreers, but they never forgot the joys of solving real world problems.')
		st.write('A few years later they decided to meet up again and started working part time on this project which they call: AI Africa.')
	

		# Building out the predication page
	if selection == "Prediction Page":

		st.markdown("![Alt Text](https://media2.giphy.com/media/k4ZItrTKDPnSU/giphy.gif?cid=ecf05e47un87b9ktbh6obdp7kooy4ish81nxm6n9c19kmnqw&rid=giphy.gif&ct=g)")
		st.info('This page uses machine learning models  to help you predict an individual\'s position  on global warming base on their tweet using')
		st.subheader('To make predictions, please follow the three steps below')
		
		#selecting input text
		text_type_selection = ['Single tweet input','multiple tweets input'] 
		text_selection = st.selectbox('Step 1 ) : Select type of tweet input', text_type_selection)

		# User selecting prediction model
		#Models = ["Logistic regression","Decision tree","Random Forest Classifier","Naive Bayes","XGboost","Linear SVC"]
		#selected_model =st.selectbox("Step 3 ) : Choose prediction model ",Models )
        

		if text_selection == 'Single tweet input':
            ### SINGLE TWEET CLASSIFICATION ###
			
            # Creating a text box for user input
			input_text = st.text_area("Step 2 ) : Enter Your Single Text Below :") 
			Models = ["Logistic regression","XGboost","Linear SVC","Random Forest"]

			selected_model = st.selectbox("Step 3 ) : Choose prediction model ",Models)

			prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
			if st.button("Classify"):
				## showing the user original text
				st.text("Input tweet is :\n{}".format(input_text))

				## Calling a function to process the text
				#tweet_text = cleaner(input_text) 

				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([input_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

            	#M Model_ Selection
				if selected_model == "Logistic regression":

					predictor = load_model("resources/Logistic_regression.pkl")
					prediction = predictor.predict(vect_text)
               	    # st.write(prediction)
				elif selected_model == "Decision tree":

					predictor = load_model("resources/Logistic_regression.pkl")
					prediction = predictor.predict(vect_text)
                    # st.write(prediction)
				elif selected_model == "Random Forest Classifier":
					predictor = load_model("resources/Logistic_regression.pkl")
					prediction = predictor.predict(vect_text)
                    # st.write(prediction)
				elif selected_model == "Naive Bayes":
					predictor = load_model("resources/Logistic_regression.pkl")
					prediction = predictor.predict(vect_text)
				elif selected_model =="XGboost" :
					 predictor = load_model("resources/Logistic_regression.pkl")
					 prediction = predictor.predict(vect_text)
				elif selected_model == "Linear SVC" :
					predictor = load_model("resources/Logistic_regression.pkl")
					prediction = predictor.predict(vect_text)
				# st.write(prediction)
			    # When model has successfully run, will print prediction
			    # You can use a dictionary or similar structure to make this output
			    # more human interpretable.
			    # st.write(prediction)
				final_result = get_keys(prediction,prediction_labels)
				st.success("Tweet Categorized as : {}".format(final_result))
    
	# Building out the "Data Visualization" page
	if selection == "Data Visualization" :

		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(train_df [['sentiment', 'message']]) # will write the df to the page
		
		# Number of Messages Per Sentiment
		st.write('Distribution of the sentiments')
        # Labeling the target
		train_df['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in train_df['sentiment']]
        
		#
		showPyplotGlobalUse = False

		# bar graph 
		colors = ['green', 'blue', 'yellow', 'red']
		sns.countplot(x='sentiment' ,data =train_df ,orient ='v',palette='PRGn')
		plt.ylabel('Count')
		plt.xlabel('Sentiment')
		plt.title('Number of Messages Per Sentiment')
		st.set_option('deprecation.showPyplotGlobalUse', False)
		st.pyplot()

		
		
		

		# Pie chart
		values = train_df['sentiment'].value_counts()/train_df.shape[0]
		labels = (train_df['sentiment'].value_counts()/train_df.shape[0]).index
		colors = ['lightgreen', 'lightblue', 'yellow', 'red']
		fig,ax=plt.subplots()
		plt.title('Distribution of ')
		ax.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=180, explode= (0.04, 0, 0, 0), colors=colors)
		st.pyplot(fig)

	   # Generating Counts of users
		st.write("Analysis of hashtags in the messages")
		train_df['users'] = [''.join(re.findall(r'@\w{,}', line)) if '@' in line else np.nan for line in train_df.message]
		counts = train_df[['message', 'users']].groupby('users', as_index=False).count().sort_values(by='message', ascending=False)
		values = [sum(np.array(counts['message']) == 1)/len(counts['message']), sum(np.array(counts['message']) != 1)/len(counts['message'])]
		labels = ['First Time Tags', 'Repeated Tags']
		colors = ['lightsteelblue', "purple"]
		fig,ax=plt.subplots()
		ax.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.04, 0), colors=colors)
		st.pyplot(fig)

        ## C
        # Popular Tags
		st.write('Popular tags found in the tweets')
		train_df['users'] = [''.join(re.findall(r'@\w{,}', line)) if '@' in line else np.nan for line in train_df.message]
		sns.countplot(y="users", hue="sentiment", data=train_df, order=train_df.users.value_counts().iloc[:10].index, palette='PRGn') 
		plt.ylabel('User')
		plt.xlabel('Number of Tags')
		plt.title('Top 20 Most Popular Tags')
		st.pyplot()
		## C






		# Popular hashtags
		st.write("The Amount of popular hashtags")
		repeated_tags_rate = round(sum(np.array(counts['message']) > 1)*100/len(counts['message']), 1)
		print(f"{repeated_tags_rate} percent of the data are from repeated tags")
		sns.countplot(y="users", hue="sentiment", data=train_df, palette='PRGn',order=train_df.users.value_counts().iloc[:10].index) 
		plt.ylabel('User')
		plt.xlabel('Number of Tags')
		plt.title('Top 20 Most Popular Tags')
		st.pyplot()


		# Generating graphs for the tags
		st.write('Analysis of most popular tags, sorted by populariy')
        # Analysis of most popular tags, sorted by populariy
		sns.countplot(x="users", data=train_df[train_df['sentiment'] == 'Positive'],order=train_df[train_df['sentiment'] == 'Positive'].users.value_counts().iloc[:20].index) 
		plt.xlabel('User')
		plt.ylabel('Number of Tags')
		plt.title('Top 20 Positive Tags')
		plt.xticks(rotation=85)
		st.pyplot()

		# Analysis of most popular tags, sorted by populariy
		st.write("Analysis of most popular tags, sorted by populariy")
		sns.countplot(x="users", data=train_df[train_df['sentiment'] == 'Negative'],
			order=train_df[train_df['sentiment'] == 'Negative'].users.value_counts().iloc[:20].index) 

		plt.xlabel('User')
		plt.ylabel('Number of Tags')
		plt.title('Top 20 Negative Tags')
		plt.xticks(rotation=85)
		st.pyplot()

		st.write("Analysis of most popular tags, sorted by populariy")
        # Analysis of most popular tags, sorted by populariy
		sns.countplot(x="users", data=train_df[train_df['sentiment'] == 'News'],
			 order=train_df[train_df['sentiment'] == 'News'].users.value_counts().iloc[:20].index) 

		plt.xlabel('User')
		plt.ylabel('Number of Tags')
		plt.title('Top 20 News Tags')
		plt.xticks(rotation=85)
		st.pyplot()

        ## C

        



    # Building the "Project team" page
	if selection == "Project Team" :
		#First row of pictures

		col1, col2,col3 = st.beta_columns(3)
		Ric_Pic =Image.open('resources/imgs/Rickie_pic.png') 
		col1.image(Ric_Pic,caption="Rickie Mogale Mohale", width=150)
		
        
		Cot_Pic =Image.open('resources/imgs/courtney_pic.png') 
		col2.image(Cot_Pic,caption="Courtney Murugan", width=150)
		

		Cot_Pic =Image.open('resources/imgs/jacques_pic.png') 
		col3.image(Cot_Pic,caption="Jacques Stander", width=150)
		

        #Second row of pictures
		col4, col5,col6 = st.beta_columns(3)
		vesh_Pic =Image.open('resources/imgs/veshen_pic.png') 
		col4.image(vesh_Pic,caption="Veshen Naidoo", width=150)
		
        
		Phiw_Pic =Image.open('resources/imgs/blue_pic.png') 
		col5.image(Phiw_Pic,caption="Phiweka Mthini", width=150)

		nor_Pic =Image.open('resources/imgs/blue_pic.png') 
		col6.image(nor_Pic,caption="Nourhan ALfalous", width=150)

		#Third row of picture 
		col7, col8,col9 = st.beta_columns(3)

		zin_Pic =Image.open('resources/imgs/zintle_pic.png') 
		col8.image(zin_Pic,caption='Zintle Faltein-Maqubela', width=150)
		col8.header("Role : Team Supervisor")
				

			

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
