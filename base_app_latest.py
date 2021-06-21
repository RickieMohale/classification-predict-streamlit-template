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
#import matplotlib.pyplot as plt
import numpy as np


from PIL import Image

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")


# defining a data Processing and cleaning  function




# The main function where we will build the actual app
def main():
	
	"""Tweet Classifier App with Streamlit """
    

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Tweet Classifer")


    #image= Image.open('resources/imgs/global_warming_img.jpg')

 


	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Background","Prediction","Data visualization", "Information","Project Team"]
	selection = st.sidebar.radio("Choose page:",options )
	#selection = st.sidebar.selectbox("Choose Option", options)

	# Builing out the "Background" page
	if selection == "Background":
		st.info('This works');	
		st.write('Explorers !!!')	
		image= Image.open('resources/imgs/global_warming_img.jpg')
		st.image(image, caption='https://www.azocleantech.com/article.aspx?ArticleID=898', use_column_width=True)
		


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
		vas_Pic =Image.open('resources/imgs/Rickie_pic.png') 
		col4.image(Ric_Pic,caption="Veshen Naidoo", width=150)
		
        
		Phiw_Pic =Image.open('resources/imgs/Rickie_pic.png') 
		col5.image(Phiw_Pic,caption="Phiweka Mthini", width=150)

		nor_Pic =Image.open('resources/imgs/Rickie_pic.png') 
		col6.image(nor_Pic,caption="Nourhan ALfalous", width=150)

		#Third row of picture 
		col7, col8,col9 = st.beta_columns(3)

		zin_Pic =Image.open('resources/imgs/zintle_pic.png') 
		col8.image(zin_Pic,caption='Zintle Faltein-Maqubela', width=150)
		col8.header("Role : Team Supervisor")
		
	# Building out the "Information" page
	if selection == "Information":

		st.info("General Information");
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here");

		st.subheader("Raw Twitter data and label");
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
	

		
		
		st.markdown("![Alt Text](https://media2.giphy.com/media/k4ZItrTKDPnSU/giphy.gif?cid=ecf05e47un87b9ktbh6obdp7kooy4ish81nxm6n9c19kmnqw&rid=giphy.gif&ct=g)")
		st.info('This page will help you predict an individual\'s position  on global warming base on their tweet')
		st.subheader('To make predictions, please follow the three steps below')
		
		#selecting input text
		text_type_selection = ['Single tweet','multiple tweets'] 
		text_selection = st.selectbox('Step 1 ) : Text input', text_type_selection)

		
		#C
		def get_keys(val,my_dict):
			for key,value in my_dict.items():
				if val == value:
					 return key
		#C


		# User selecting prediction model
		#Models = ["Logistic regression","Decision tree","Random Forest Classifier","Naive Bayes","XGboost","Linear SVC"]
		#selected_model =st.selectbox("Step 3 ) : Choose prediction model ",Models )
        

		if text_selection== 'Single tweet':
            ### SINGLE TWEET CLASSIFICATION ###
			
            # Creating a text box for user input
			input_text = st.text_area("Step 2 ) : Enter Your Single Text Below :") ##user entering a single text to classify and predict
			Models = ["Logistic regression","Decision tree","Random Forest","Naive Bayes","XGboost","Linear SVC" ]
			selected_model = st.selectbox("Step 3 ) : Choose prediction model ",Models)
			def load_prediction_models(model_file):

				loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
				return loaded_models

			prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1}
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

					predictor = load_prediction_models("resources/Logistic_regression.pkl")
					prediction = predictor.predict(vect_text)
               	    # st.write(prediction)
				elif selected_model == "Decision tree":

					predictor = load_prediction_models("resources/Logistic_regression.pkl")
					prediction = predictor.predict(vect_text)
                    # st.write(prediction)
				elif selected_model == "Random Forest Classifier":
					predictor = load_prediction_models("resources/Logistic_regression.pkl")
					prediction = predictor.predict(vect_text)
                    # st.write(prediction)
				elif selected_model == "Naive Bayes":
					predictor = load_prediction_models("resources/Logistic_regression.pkl")
					prediction = predictor.predict(vect_text)
				elif selected_model =="XGboost" :
					 predictor = load_prediction_models("resources/Logistic_regression.pkl")
					 prediction = predictor.predict(vect_text)
				elif selected_model == "Linear SVC" :
					predictor = load_prediction_models("resources/Logistic_regression.pkl")
					prediction = predictor.predict(vect_text)
				# st.write(prediction)
			    # When model has successfully run, will print prediction
			    # You can use a dictionary or similar structure to make this output
			    # more human interpretable.
			    # st.write(prediction)
				final_result = get_keys(prediction,prediction_labels)
				st.success("Tweet Categorized as : {}".format(final_result))


				

			

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
