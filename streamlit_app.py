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
import plotly.figure_factory as ff
import plotly.graph_objects as go
import base64


import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
_lock = RendererAgg.lock
#suppress cell_warnings
import warnings
warnings.filterwarnings("ignore")


# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load data 
train_df = pd.read_csv("resources/train.csv")

# defining a data Processing and cleaning  function
def tweet_processor(input_t):


    # Removing https and trailing white spaces
	input_t = re.sub(r'^RT ','', re.sub(r'https://t.co/\w+', '', input_t).strip()) 

    # Removing puctuations
	input_tweet = input_t.lower()
	tweet = ''.join([x for x in input_t if x not in string.punctuation])

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


# Prediction classification function
def predict_class(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key
# The main function where we will build the actual app
def main():
	
	"""Tweet Classifier App with Streamlit """

    ### Loading Company logo
	row1_space1, center_, row1_space2 = st.beta_columns((.5, 1, .2, ))
	with center_,_lock :

		file_ = open('resources/imgs/Company_logo.gif', "rb")
		contents = file_.read()
		data_url = base64.b64encode(contents).decode("utf-8")
		file_.close()
		st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True,)
	


	pages = ["Prediction Page","Data Visualization", "Company Information, Background & Team"]
	selection = st.sidebar.selectbox("Choose Page :", pages)

	# Building out the "Background" page
	if selection == "Company Information, Background & Team":
		st.title("1Company Information, Background and Team")
		st.info('Discover the mission and vision that keeps us going as well as the amazing team that pulled this project together and how we started.')

		st.header('Our Mission')		
		st.write('To use AI to combat climate change within Africa, securing the futures of the generations of now and tomorrow.')

		st.header('Our Vision')
		st.write('A better and more intelligent Africa which is able to adapt to the fourth industrial revolution by using Data Science, for social good.')

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
        
		Phiw_Pic =Image.open('resources/imgs/phiwe_pic.png') 
		col5.image(Phiw_Pic,caption="Phiweka Mthini", width=150)
		col5.write('Digital marketer ')

		nor_Pic =Image.open('resources/imgs/nour_pic.png') 
		col6.image(nor_Pic,caption="Nourhan ALfalous", width=150)
		col6.write('Database architect')

		#Third row of picture 
		col7, col8,col9 = st.beta_columns(3)

		st.header('How we started?')
		st.write('African Intelligence started as a group of 6 students who met each other on a university project. The students bonded together around a love for solving problems with the help of AI. ')	
		st.write('These students all graduated with flying colours and entered succesful carreers, but they never forgot the joys of solving real world problems.')
		st.write('A few years later they decided to meet up again and started working part time on this project which they call: AI Africa.')
	

		# Building out the predication page
	if selection == "Prediction Page":

		
		row1_space1, center_, row1_space2 = st.beta_columns((.5, 1, .2, ))
		with center_,_lock :
			st.subheader('Sentiment Prediction Page')
		

		
		st.info('This page uses machine learning models  to help you predict an individual\'s position  on global warming base on their tweet')
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
			Models = ["Logistic Regression","Linear SVC","Naive Bayes multinomial","Ridge classifier"]

			selected_model = st.selectbox("Step 3 ) : Choose prediction model ",Models)

			prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
			if st.button("Classify"):
				## showing the user original text
				st.text("Input tweet is :\n{}".format(input_text))

				## Calling a function to process the text
				#tweet_text = cleaner(input_text) 

				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([input_text]).toarray()
				

            	#M Model_ Selection
				if selected_model == "Logistic Regression":

					predictor = load_model("resources/logreg_count2.0.pickle")
					prediction = predictor.predict(vect_text)
               	    # st.write(prediction)
				elif selected_model == "Linear SVC":

					predictor = load_model("resources/svm_model.pkl")
					prediction = predictor.predict(vect_text)
                    # st.write(prediction)
				elif selected_model == "Naive Bayes multinomial":
					predictor = load_model("resources/nbm_count.pkl")
					prediction = predictor.predict(vect_text)
                    # st.write(prediction)
				elif selected_model == "Ridge classifier":
					predictor = load_model("resources/ridge_count2.0.pickle")
					prediction = predictor.predict(vect_text)

				# st.write(prediction)
			    # When model has successfully run, will print prediction
			    # You can use a dictionary or similar structure to make this output
			    # more human interpretable.
			    # st.write(prediction)
				final_result = predict_class(prediction,prediction_labels)
				st.success("Tweet Categorized as : {}".format(final_result))
			st.markdown("![Alt Text](https://media2.giphy.com/media/k4ZItrTKDPnSU/giphy.gif?cid=ecf05e47un87b9ktbh6obdp7kooy4ish81nxm6n9c19kmnqw&rid=giphy.gif&ct=g)")
    
	# Building out the "Data Visualization" page
	if selection == "Data Visualization" :

		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(train_df [['sentiment', 'message']]) # will write the df to the page
		

        # Labeling the target
		train_df['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in train_df['sentiment']]
        
		#


		showPyplotGlobalUse = False

		

		## plottin pie chart side by side
		st.write('')
		row1_space1, row1_1, row1_space1, row1_2, row1_space1 = st.beta_columns((.1, 1, .1, 1, .1))

		#first pie chart
		with row1_1, _lock:

			st.subheader("Distribution Of Sentiments")
			values = train_df['sentiment'].value_counts()/train_df.shape[0]
			labels = (train_df['sentiment'].value_counts()/train_df.shape[0]).index
			colors = ['lightgreen', 'lightblue', 'yellow', 'red']
			fig1,ax=plt.subplots()
			#plt.title('Distribution Of Sentiments ')
			ax.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=180, explode= (0.04, 0, 0, 0), colors=colors)
			st.pyplot(fig1)
			st.markdown("This pie chart show\'s that majority of people believe that global warming is real")

			
        #second pie chart
		with row1_2, _lock:
			
			st.subheader("Reapeted  Vs First Time Tags")
			train_df['users'] = [''.join(re.findall(r'@\w{,}', line)) if '@' in line else np.nan for line in train_df.message]
			counts = train_df[['message', 'users']].groupby('users', as_index=False).count().sort_values(by='message', ascending=False)
			values = [sum(np.array(counts['message']) == 1)/len(counts['message']), sum(np.array(counts['message']) != 1)/len(counts['message'])]
			labels = ['First Time Tags', 'Repeated Tags']
			colors = ['lightgreen', "lightblue"]
			fig2,ax=plt.subplots()
			ax.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=180, explode= (0.04, 0), colors=colors)
			st.pyplot(fig2)
			st.write("This pie chart show\'s that they are specific people or entities who are frequently tagged about global warming")



 		## 'Number of Tweets Per Sentiment' bargraph
		row1_space1, center_, row1_space2 = st.beta_columns((.5, 1, .2, ))
		with center_,_lock :
			st.subheader('Number of Tweets Per Sentiment')


		
		fig3 =Figure()
		ax = fig3.subplots()
		colors = ['green', 'blue', 'yellow', 'red']
		sns.countplot(x='sentiment' ,data =train_df ,palette='PRGn',ax=ax)
		ax.set_ylabel('Number Of Tweets')
		plt.title('Number of Tweets Per Sentiment')
		st.pyplot(fig3)
		st.write("")

		## Plotting   'Top 10 People or entities mostly tagged about global warming ' 
		row1_space1, center_, row1_space2 = st.beta_columns((.2, 1, .2, ))

		with center_,_lock :
			st.subheader( 'Top 10 of People or entities mostly tagged about global warming ')    
		

		fig4 =Figure()
		ax = fig4.subplots()
		train_df['users'] = [''.join(re.findall(r'@\w{,}', line)) if '@' in line else np.nan for line in train_df.message]
		sns.countplot(y="users", hue="sentiment", data=train_df, order=train_df.users.value_counts().iloc[:10].index, palette='PRGn',ax=ax) 
		plt.ylabel('People or Entities tagged')
		plt.xlabel('Total Number Of Tags')
		st.pyplot(fig4)
		
        

		## Plotting mostly tagged people about global warming ' 
		row1_space1, row1_1, row1_space1, row1_2, row1_space1 = st.beta_columns((.1, 1, .1, 1, .1))
		with row1_1, _lock:
			st.subheader('Top 10 people or entities  tagged about posative sentiments')
			fig5 =Figure()
			ax = fig5.subplots()
			sns.countplot(y="users", data=train_df[train_df['sentiment'] == 'Positive'],order=train_df[train_df['sentiment'] == 'Positive'].users.value_counts().iloc[:10].index,ax=ax) 
			plt.ylabel('Total Number Of Tags')
			st.pyplot(fig5)

		with row1_2, _lock:
			st.subheader("Top 10 people or entities  tagged about negative sentiments")
			fig6 =Figure()
			ax = fig6.subplots()
			sns.countplot(y="users", data=train_df[train_df['sentiment'] == 'Negative'],order=train_df[train_df['sentiment'] == 'Negative'].users.value_counts().iloc[:10].index,ax=ax) 
			plt.ylabel('Total Number Of Tags')
			st.pyplot(fig6)

		row1_space1, center_, row1_space2 = st.beta_columns((.2, 1, .2, ))
		with center_,_lock :
			st.subheader( 'Top 10 News Outlets tagged about global warning') 
			fig7 =Figure()
			ax = fig7.subplots()
			sns.countplot(y="users", data=train_df[train_df['sentiment'] == 'News'],order=train_df[train_df['sentiment'] == 'News'].users.value_counts().iloc[:10].index,ax=ax) 
			plt.xlabel('User')
			plt.ylabel('Total Number Of Tags')
			st.pyplot(fig7)




       

		# showing posative tweets
		#corpus = re.sub("climate change", ''," ".join(tweet.strip() for tweet in train_df['clean'][working_df['sentiment'] == 'Positive']))
		#wordcloud = WordCloud(font_path='../input/droidsansmonottf/droidsansmono.ttf', background_color="white",width = 1920, height = 1080, colormap="viridis").generate(corpus)
		#plt.figure(dpi=260)
		#plt.imshow(wordcloud, interpolation='bilinear')
		#plt.axis("off")
		#plt.show()








        ## C

        





			

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
