import streamlit as st
import altair as alt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import normaltest
import nltk
from collections import Counter
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import gensim
import string
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
import spacy_streamlit as spat 
from spacy import displacy
from pprint import pprint 
from wordcloud import WordCloud
import geopandas as gpd
from PIL import Image
import time
import pydeck as pdk
import cufflinks as cf
from plotly.offline import init_notebook_mode, plot, iplot
import pickle
from tokenization import tokenizer
import os
sns.set(style='whitegrid')

#to get connection
st.set_page_config(
page_title="CMR Conflict DM App",
page_icon= ":smiley:",
layout="centered",
initial_sidebar_state="expanded")

init_notebook_mode(connected=True)
cf.go_offline(connected=True)

file = '/home/massock/Images/screenshot-looka.com-2020.06.27-13_47_43.png'
image = Image.open(file)
img= st.sidebar.image(image, use_column_width=True)


#front end elements of web page
html_page = """
<div style ="background-color:green;padding:23px>
<h1 style ="color:black;text-align:center;"> CMR Conflict Data Mining App</h1>
"""

#display the front end aspect 
st.markdown(html_page, unsafe_allow_html=True)

st.sidebar.header('****** About CMR Conflict DM ******')
st.sidebar.text(""" 
CMR Conflict DM is a Data Mining 
Board app which allows an user to
understand and discover a knownledge 
in the conflict database for Cameroon.
Be free to enjoy this app!.
    """)

#title
st.title("Conflict in Cameroon")

#description
text = """    
Conflict data for Cameroon come from ACLED (Armed Conflict Location & Event Data Project) 
that report information on the type, agents, exact location, date, and other 
characteristics of political violence events, demonstrations and select political
relevant non-violent events.

For more information click [here](https://acleddata.com/).
"""
st.markdown(text, unsafe_allow_html=True)

#change path 
conflict = pd.read_excel('conflict_data_cmr.xlsx')

if st.checkbox('show/hide data', key=0):

	#show data
	st.dataframe(conflict)

	#display information
	info = """

	**Feature explanation** 

	1. **event_date**: The	day,month and year on which an event took place.
	2. **event_type**: The type of event.
	3. **sub_event_type**: The type of sub event of event.
	4. **actor1**: The named actor	involved in	the	event. 
	5. **actor2**: The named actor involved in the event.
	6. **interaction**: A numeric code indicating the interaction between types of ACTOR1 and ACTOR2.
	7. **region**: The region of the world where the event took place.
	8. **admin1**: The largest sub-national administrative region in which the event took place.
	10. **admin2**: The	second largest sub-national	administrative region in which the event took place
	11. **admin3**: The third largest sub-national administrative region in which the event took place.
	12. **location**: The location in which	the	event took place.
	13. **latitude**: The latitude of the location. 
	14. **longitude**: The longitude of the location
	15. **source**: The	source of the event report.
	16. **source_scale**: The scale (local, regional, national, international) of the source
	17. **notes**:A	short description of the event
	18. **fatalities**:The number of reported fatalities which occurred during the event

	a. For more detail click [here](https://acleddata.com/#/dashboard)

	b. For codebook ACLED click [here](https://reliefweb.int/sites/reliefweb.int/files/resources/ACLED_Codebook_2017FINAL%20%281%29.pdf)

	Now, we can clean and prepare our data for Exploratory data analysis. 

	**N.B**: cmr conflict data go from 1997 to 2020.

	"""

	with st.beta_expander('data information'):
		st.markdown(info, unsafe_allow_html=True)

#clean and prepare data
@st.cache
def clean_prepare():
	data_conflict = conflict.iloc[1:, :31]

	data = data_conflict.drop(columns=['iso', 'event_id_cnty', 'event_id_no_cnty', 'assoc_actor_1',\
                                   'assoc_actor_2', 'geo_precision', 'iso3', 'data_id', 'time_precision',
                                   'inter1', 'inter2', ])

	data = data.fillna(' ')

	data['event_date'] = pd.to_datetime(data['event_date'])
	data['year'] = pd.to_numeric(data['year'], errors='coerce')
	data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
	data['longitude'] = pd.to_numeric(data['longitude'],errors='coerce')
	data['fatalities'] = pd.to_numeric(data['fatalities'], errors='coerce')
	data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

	return data

st.sidebar.header('Exploratory data analysis')
if st.sidebar.checkbox('show cleaning & preparing', key=0):
	st.header('Clean & prepare data')
	st.dataframe(clean_prepare())
	with st.beta_expander('Learn more'):
		st.markdown("""We take only relevant informative features, make missing value imputation 
		and change dtype of certains feature. 
		 """)


#distribution visualization
if st.sidebar.checkbox('Distribution visualization', key=2):
	st.header('Distribution visualization')
	data = clean_prepare()

	#button dist
	with st.beta_container():

		if st.sidebar.button('Administrative region conflict', key=1):
			fig, ax =plt.subplots()
			data.admin1.value_counts().plot(kind='bar')
			ax.set_title('Administrative region where exist different conflict.')
			ax.set_ylabel('count')
			st.pyplot(fig)	
			with st.beta_expander("Learn more"):
				st.markdown("""Cameroon and Middle Africa can be taken respectively as an administrative
		 	region of Middle Africa and Africa.  """)


		if st.sidebar.button('Most common Department conflict', key=2):
			fig, ax =plt.subplots()
			data.admin2.value_counts()[:20].plot(kind='bar')
			ax.set_title('Most common Department of Cameroon where there exist conflict.')
			ax.set_ylabel('count')
			st.pyplot(fig)
			with st.beta_expander("Learn more"):
				st.markdown(""" We take only 20 most commons Department.  """)

		if st.sidebar.button('Most common Division conflict',key=3):
			fig, ax =plt.subplots()
			data.admin3.value_counts()[:20].plot(kind='bar')
			ax.set_ylabel('count')
			ax.set_title('Most common Division where occurs the conflict in Cameroon')
			st.pyplot(fig)
			with st.beta_expander("Learn more"):
				st.markdown(""" We take only 20 most commons Division.  """)

		if st.sidebar.button('Most common location conflict', key=4):
			fig, ax =plt.subplots()
			data.location.value_counts()[:20].plot(kind='bar')
			ax.set_ylabel('count')
			ax.set_title('Most common location where occurs a conflict.')
			st.pyplot(fig)
			with st.beta_expander("Learn more"):
				st.markdown(""" We take only 20 most commons Location.  """)

		if st.sidebar.button('Most common source media', key=5):
			fig, ax =plt.subplots()
			data.source.value_counts()[:20].plot(kind='bar')
			ax.set_ylabel('count')
			ax.set_title('Most common the different sources media.')
			st.pyplot(fig)
			with st.beta_expander("Learn more"):
				st.markdown(""" We take only 20 most commons sources media.  """)


		if st.sidebar.button('Most common source scale', key=6):
			fig, ax =plt.subplots()
			data.source_scale.value_counts()[:20].plot(kind='bar')
			ax.set_title('Source scale')
			ax.set_ylabel('count')
			st.pyplot(fig)
			with st.beta_expander("Learn more"):
				st.markdown(""" We take only 20 most commons source scale.  """)

		if st.sidebar.button("Conflict event type", key=7):
			fig, ax = plt.subplots()
			data.event_type.value_counts().plot(kind='bar')
			ax.set_title('The event type of conflict in Cameroon.')
			ax.set_ylabel('count')
			st.pyplot(fig)

	if st.sidebar.checkbox('Conflict sub event type', key=1):
		st.subheader('Conflict sub event.')
		sub_event = dict()
		for u in data.event_type.unique():
		    sub_event[u] = data[data.event_type == u].sub_event_type.value_counts()

		u = st.selectbox('Select event type:' , tuple(sub_event.keys()))

		figure, ax = plt.subplots() 
		sub_event[u].plot(kind='bar', ax=ax)
		ax.set_ylabel('counts')
		ax.set_title(f'Event conflict type: {u}.')
		st.pyplot(figure)
		with st.beta_expander("Learn more"):
			st.markdown(""" Along the x-axis we have a sub event type, y-axis a counts.   """)


	if st.sidebar.button('Interaction', key=9):
		fig, ax = plt.subplots()
		data.interaction.value_counts()[:20].plot(kind='bar')
		ax.set_title('Interacton distribution')
		ax.set_ylabel('counts')
		st.pyplot(fig)
		with st.beta_expander("Learn more"):
			st.markdown(""" For more detail click 
			[codebook ACLED](https://reliefweb.int/sites/reliefweb.int/files/resources/ACLED_Codebook_2017FINAL%20%281%29.pdf)
			  """)

	if st.sidebar.checkbox('Jointplot lat & long', key=2):
		st.subheader('latitude & longitude')
		fig, ax = plt.subplots(1,1)
		sns.jointplot(x='longitude', y='latitude', data=data, kind='kde', ax=ax)
		ax.set_title('Cameroon conflict density')
		ax.set_xlabel('longitude')
		ax.set_ylabel('latitude')
		st.pyplot(fig)
		with st.beta_expander('Learn more'):
			st.markdown("""
			Jointplot draws a plot of two variables with bivariate and univariate graphs.
			 """)

		if st.sidebar.button('Distplot', key=11):
			figureb, (ab1,ab2) = plt.subplots(1,2)
			sns.distplot(data.latitude, ax=ab1)
			sns.distplot(data.longitude,  ax=ab2)
			st.pyplot(figureb)
			with st.beta_expander('Learn more'):
				st.markdown("""
				Flexibly plot a univariate distribution of observations.
				 """)

		if st.sidebar.button('Boxplot', key=12):
			figure, (a1,a2,a3) = plt.subplots(1,3)
			sns.boxplot(x='latitude', data=data, ax=a1)
			sns.boxplot(x='longitude', data=data, ax=a2)
			sns.boxplot(x='fatalities', data=data, ax=a3)
			st.pyplot(figure)
			
			with st.beta_expander('Learn more'):
				st.markdown("""
				Draw a box plot to show distributions with respect to categories.
				 """)

#discriptive analysis
if st.sidebar.checkbox('Descriptive analysis', key=3):
	data = clean_prepare()

	st.header("Descriptive analysis")
	if st.sidebar.button('Describe data', key=14):
		st.subheader('Describe data')
		st.dataframe(data.describe())

	
	if st.sidebar.checkbox('correlation', key=4):
		st.subheader('Correlation')
		st.dataframe(data.corr())
		with st.beta_expander('Learn more'):
			st.markdown(""" 
			Two variables are positive correlated if only if

			1. $corr >= 0.5$.
			2. $corr >= 0.8$ (strong).

			Two variables are negative correlated if only if

			1. $corr < -0.5$.
			2. $corr <= -0.8$ (strong).

			Assume that $x$ and $y$ are two independent variables; if we compute $corr(x,y) >= 0.5$ or $corr(x,y) < -0.5$ or 
			$corr(x,y) >= 0.8$ or $corr(x,y)<= -0.8$,
			we can plot this
			> $y = f(x)$ this means that the trend of $y$ depends on trend of $x$. 

			**N.B**: 

			1. Positive correlation means that $x$ and $y$ go together i.e if $x$ increase over time, $y$ increase over time.
			2. Negative correlation means that $x$ and $y$ does not go together if $x$ increase over time, $y$ decrease over time.

			""")

		year = st.selectbox('Select year:', sorted(list(data.year.unique())))

		#correlation for each year
		result = 'corr(latitude, longitude) = {}.'.format(data[data.year==year].corr().loc['latitude', ['longitude']].values[0])
		st.success(result)

		fig, ax = plt.subplots()
		sns.regplot(x='longitude', y='latitude', data=data[data.year==year], lowess=True)
		title = 'Regression plot latitude and longitude for year {}'.format(year)
		ax.set_title(title)
		st.pyplot(fig)


		cols = ['event_date','admin1','location','event_type','sub_event_type','fatalities']
		if st.checkbox('Geolocalization',key=5):
			st.subheader('Geolocalization')
			st.dataframe(data[data.year==year][cols])
			st.dataframe(data[data.year==year][['event_date','location','notes', 'fatalities']])

		@st.cache
		def curiosity():

			#initialize
			corr = []
			years = []
			total_fata = []
			admin1 = []
			ev_tpe = []
			sub_type = []

			for u in sorted(list(data.year.unique())):

				corr.append(data[data.year==u].corr().loc['latitude', ['longitude']].values[0])
				years.append(u)
				admin1.append(data[data.year==u].admin1.mode().values)
				total_fata.append(data[data.year==u].fatalities.sum())
				ev_tpe.append(data[data.year==u].event_type.mode().values)
				sub_type.append(data[data.year==u].sub_event_type.mode().values)

			cdata = pd.DataFrame()

			cdata['corr(lat,long)'] = corr
			cdata['year'] = years
			cdata['total_fatalities'] = total_fata
			cdata['admin1_mode'] = admin1
			cdata['event_type_mode'] = ev_tpe
			cdata['sub_event_type_mode'] = sub_type

			return cdata

		if st.sidebar.checkbox('Some curiosity',key=6):
			st.subheader('Relevant informative data')
			df = curiosity()
			cd = df.set_index('year')
			st.dataframe(df)

			if st.button('plot corr(lat,long) vs total_fatalities'):
				c = alt.Chart(df).mark_bar().encode(x='corr(lat,long)', y='total_fatalities', 
					tooltip=['corr(lat,long)', 'total_fatalities'])
				st.altair_chart(c, use_container_width=True)

			if st.button('Heatmap calendar'):
				
				fig, ax = plt.subplots()
				sns.heatmap(cd[['corr(lat,long)', 'total_fatalities']],center=0, annot=True, fmt='.6g')
				ax.set_title('Heatmap calendar.')
				st.pyplot(fig)

	#conflict is spreading
	if st.sidebar.checkbox('is conflict spreading?', key=7):
		st.subheader('Conflict is spreading.')
		year_fata = data.groupby('year')['fatalities'].agg('sum')
		#event type section
		if st.sidebar.checkbox('event type', key=8):
			st.subheader('Event type')
			if st.sidebar.button('fatalities barplot'):
				fig, ax = plt.subplots()
				year_fata.plot(kind='bar')
				ax.set_ylabel('cummulative fatalities')
				ax.set_title('Progresssive of fatalities caused by conflict in Cameroon.')
				st.pyplot(fig)

			event_conflict = pd.pivot_table(data, values='fatalities', 
					columns='event_type', index='year', aggfunc='sum')

			if st.sidebar.button('calendar event type', key=15):
				fig, ax = plt.subplots()
				sns.heatmap(event_conflict, center=0, annot=True, fmt='.6g')
				ax.set_title('Heatmap of conflict in Cameroon.')
				st.pyplot(fig)

				with st.beta_expander('Learn more'):
					st.markdown("""
					The blank space means that no data are recorded in that year corresponding to the event type. 
					 """)

			if st.sidebar.button('event type describe', key=16):
				st.dataframe(event_conflict.describe())

			if st.sidebar.button('event type similarity', key=17):
				st.dataframe(event_conflict.corr())
				with st.beta_expander('Learn more'):
					st.markdown("""
					Refer to correlation learn more section.
					 """)

			if st.sidebar.button('sub event similarity', key=18):
				sub_conflict = pd.pivot_table(data, values='fatalities', index='year',
				 columns='sub_event_type', aggfunc='sum')

				st.dataframe(sub_conflict.corr())
				with st.beta_expander('Learn more'):
					st.markdown("""
					NaN value means that correlation  
					 """)

		# Administrative region
		if st.sidebar.checkbox('conflict administrative region', key=9):
			st.subheader('Conflict administrative region')
			region = pd.pivot_table(data, values='fatalities', columns='admin1',
			 index='year', aggfunc='sum')

			if st.sidebar.button('fatalities calendar'):
				fig, ax = plt.subplots()
				sns.heatmap(region, annot=True, fmt='.4g')
				ax.set_title('Fatalities calandar conflict in Cameroon ')
				st.pyplot(fig)
				with st.beta_expander('Learn more'):
					st.markdown("""
					Do not forget that Cameroon is taken like administrative region for Middle Africa and Midlle Africa 
					as administrative region of Africa.
					And also, some conflict are near to the boundary of Cameroon.
					  """)

			if st.sidebar.button('conflict describe'):
				st.dataframe(region.describe())

			if st.sidebar.button('conflict similarity'):
				st.dataframe(region.corr())

				with st.beta_expander('Learn more'):
					st.markdown("""
					correlation give similarity between two variables for data going to 1997 to 2020.
					  """)

			if st.sidebar.button('noso 2016-2020 similarity'):

				noso = region[region.index.isin(['2016','2017','2018','2019','2020'])].corr()

				fig, ax = plt.subplots()
				sns.heatmap(noso, annot=True, fmt='.1g')
				ax.set_title('4 years of noso conflict similarity.')
				st.pyplot(fig)
				with st.beta_expander('Learn more'):
					st.markdown("""
					This part shows how 4 years of war in noso region is similar to the conflicts in other
					 regions of cameroon and central Africa.
					  """)

#NLP
st.sidebar.header('Text mining')
if st.sidebar.checkbox('Natural language processing', key=10):
	st.header('Natural language processing')

	#create function tokenizer
	def sentence_tokenizer(sentence):
	    from spacy.lang.en.stop_words import STOP_WORDS
	    from spacy.lang.en import English
	    
	    # Create our list of punctuation marks
	    punctuations = string.punctuation

	    # Create our list of stopwords
	    nlp = spacy.load('en_core_web_sm')
	    stop_words = spacy.lang.en.stop_words.STOP_WORDS

	    # Load English tokenizer, tagger, parser, NER and word vectors
	    parser = English()
	    # Creating our token object, which is used to create documents with linguistic annotations.
	    mytokens = parser(sentence)

	    # Lemmatizing each token and converting each token into lowercase
	    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]

	    # Removing stop words
	    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

	    # return preprocessed list of tokens
	    return mytokens

	#compute word frequency
	@st.cache
	def word_frequence(dtext, n=1):
		
		tfvector = TfidfVectorizer(tokenizer=sentence_tokenizer, ngram_range=(n, n))
		transformed_text = tfvector.fit_transform(dtext)
		transformed_text_as_array = transformed_text.toarray()

		for counter, doc in enumerate(transformed_text_as_array):
        	#construct a dataframe
			tf_idf_tuples = list(zip(tfvector.get_feature_names(), doc))
			one_doc_as_df = pd.DataFrame.from_records(tf_idf_tuples, 
			columns=['term', 'score']).sort_values(by='score', ascending=False).reset_index(drop=True)

		return one_doc_as_df

	data = clean_prepare() #load data 
	if st.sidebar.checkbox('Text mining'):
		st.subheader('Text exploration')

		year = st.sidebar.selectbox('Select conflict notes by year:', tuple(sorted(data.year.unique())))
		default_text = data[data.year == year][['event_date', 'notes', 'source', 'source_scale']].sort_values(by='event_date', 
		ascending=True).reset_index(drop=True)

		default_text['month'] = default_text.event_date.dt.month# create month variable

		st.write(default_text,  unsafe_allow_html=True) # display

		month = st.selectbox('Choose month available:', list(default_text.month.unique()))

		#select notes by month
		text_data = default_text[default_text.month == month][['event_date', 'notes', 'source', 'source_scale']]
		dict_month = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'Juillet', 8:'August',
		9:'Septembre', 10:'October', 11:'November', 12:'December'}

		#display text 
		st.write('Notes for', dict_month[month],':', text_data, unsafe_allow_html=True)

		#start text analysis
		if st.checkbox('Text analysis'):
			notes = text_data.notes.values
			#joint all sentence to one
			join_sent = " ".join(u for u in notes)
			
			st.subheader('Word Frequency')
			with st.beta_expander('Learn more'):
				st.markdown("""
					**Word Frequency** is a text analysis technique that measures the most frequently 
					occuring words or concept in given text using the numerical statistics **TF-IDF**(Term
					Frequency-Inverse Document Frequency)
					""")
			if st.button('word frequency', key=0):
				st.write(word_frequence(notes))

			st.subheader('Collocations')
			n_grams = st.selectbox('Choose n-gram', [2,3])
			if st.button('n-grams:'):
				st.write(word_frequence(notes, n=n_grams))
			with st.beta_expander('Learn more'):
				st.markdown("""
					**Collocations** helps identify words that commonly co-occur.(bigrams, trigrams, ...)
					""")

			st.subheader("WordCloud or Keyword Extraction")

			with st.beta_expander('Learn more'):
				st.markdown("""
					**Word Cloud** is a visual representation of words. is also a technique to show which 
					words are the most frequent among the given text.
					""")
			#wordcloud	
			if st.button('word cloud'):
				from spacy.lang.en.stop_words import STOP_WORDS
				stop_words = spacy.lang.en.stop_words.STOP_WORDS

				#joint all token in one sentence.
				token_join = " ".join(u for u in sentence_tokenizer(join_sent))

				#create wordcloud
				wordcloud = WordCloud(stopwords=stop_words).generate(token_join)
				fig, ax = plt.subplots()
				ax.imshow(wordcloud, interpolation='bilinear')
				ax.axis("off")
				st.write(fig)

			#concordance
			st.subheader('Concordance')
			with st.beta_expander('Learn more'):
				st.markdown("""
				**Concordance** helps identify the context and instances of words or a set of words.
				""")

			#word = st.text_input("Give a word:")
			#concordance
			#if st.button('concordance'):
				#tokens = nltk.word_tokenize(join_sent)
				#text = nltk.Text(tokens)
				#conc = text.concordance('village')
				#st.write(conc)

			st.subheader('Name Entity Recognition')
			with st.beta_expander('Learn more'):
				st.markdown("""
					**NER** find entities, which can be people, companies, or locations and exist within text data. 
					""")
			nlp = spacy.load('en')
			doc = nlp(join_sent)
			if st.button('NER'):
				spat.visualize_ner(doc, labels=nlp.get_pipe('ner').labels)


#Geospatial analysis
st.sidebar.header('Geospatial analysis')
if st.sidebar.checkbox('Geospatial analysis'):
	st.subheader('Geospatial analysis')
	data = clean_prepare()

	year = st.sidebar.selectbox('Select year conflict:', list(sorted(data.year.unique())))
	year_data = data[data.year == year]

	st.map(year_data[['latitude', 'longitude']], zoom=6)


	if st.sidebar.checkbox('Admin barh'):
    		
		st.subheader('Administrative fatalities')

		fig, ax = plt.subplots()

		ax.barh(year_data.admin1, year_data.fatalities)
		ax.set_xlabel('Fatalities')
		ax.set_ylabel('Administrative')
		ax.set_title(f'{year} Geospatial fatalities.')
		st.pyplot(fig)

#time series
st.sidebar.header('Time series')
if st.sidebar.checkbox('Time series'):
	st.subheader('Time series')
	data = clean_prepare()

	data = data.set_index('event_date')

	st.write(data)
	if st.checkbox('All time series'):
		with st.beta_container():

			st.subheader('Daily fatalities') 
			figd, axd = plt.subplots()
			axd = data.fatalities.resample('D').sum().plot()
			axd.set_ylabel('fatalities')
			st.pyplot(figd)

			st.subheader('Weekly fatalities') 
			figw, axw = plt.subplots()
			axw = data.fatalities.resample('W').sum().plot()
			axw.set_ylabel('fatalities')
			st.pyplot(figw)

			st.subheader('Monthly fatalies') 
			figm, axm = plt.subplots()
			axm = data.fatalities.resample('M').sum().plot()
			axm.set_ylabel('fatalities')
			st.pyplot(figm)

			st.subheader('Yearly fatalities') 
			figy, axy = plt.subplots()
			axy = data.fatalities.resample('Y').sum().plot()
			axy.set_ylabel('fatalities')
			st.pyplot(figy)

			st.subheader('Quarterly fatalities') 
			figq, axq = plt.subplots()
			axq = data.fatalities.resample('Q').sum().plot()
			axq.set_ylabel('fatalities')
			st.pyplot(figq)

	if st.sidebar.checkbox('Case study 2020'):
		dat_2020 = data[data.year == 2020][['latitude', 'longitude', 'fatalities']]

		st.subheader('Case study 2020.')
		st.write(dat_2020)
		#case study

		name = st.selectbox('Select feature:', list(dat_2020.columns))

		st.write('Boxplot')
		fig, ax = plt.subplots()
		sns.boxplot(dat_2020[name])
		st.pyplot(fig)

		if st.button('correlation'):
			st.dataframe(dat_2020.corr())

		if st.button('time series 2020', key=1):
			
			with st.beta_container():
				st.subheader('Daily fatalities') 
				figd, axd = plt.subplots()
				axd = dat_2020.fatalities.resample('D').sum().plot()
				axd.set_ylabel('fatalities')
				st.pyplot(figd)

				st.subheader('Weekly fatalities') 
				figw, axw = plt.subplots()
				axw = dat_2020.fatalities.resample('W').sum().plot()
				axw.set_ylabel('fatalities')
				st.pyplot(figw)

				st.subheader('Monthly fatalities') 
				figm, axm = plt.subplots()
				axm = dat_2020.fatalities.resample('M').sum().plot()
				axm.set_ylabel('fatalities')
				st.pyplot(figm)

				st.subheader('Quarterly fatalities') 
				figq, axq = plt.subplots()
				axq = dat_2020.fatalities.resample('Q').sum().plot()
				axq.set_ylabel('fatalities')
				st.pyplot(figq)

				st.subheader('longitude time series') 
				figq, axq = plt.subplots()
				axq = dat_2020.longitude.plot()
				axq.set_ylabel('longitude')
				st.pyplot(figq)

		if st.button("fatalities seasonality", key=2):

			
			decompose_fata = sm.tsa.seasonal_decompose(dat_2020.fatalities, period=12)
			

			with st.beta_container():
				st.subheader('trend')
				fig, ax = plt.subplots()
				decompose_fata.trend.plot()
				st.pyplot(fig)

				st.subheader('seasonal')
				fig, ax = plt.subplots()
				decompose_fata.seasonal.plot()
				st.pyplot(fig)
				
				st.subheader('residual')
				fig, ax = plt.subplots()
				decompose_fata.resid.plot()
				st.pyplot(fig)


		if st.button("longitude seasonality",key=3):

			decompose_long = sm.tsa.seasonal_decompose(dat_2020.longitude, period=12)
			with st.beta_container():
				st.subheader('trend')
				fig, ax = plt.subplots()
				decompose_long.trend.plot()
				st.pyplot(fig)

				st.subheader('seasonal')
				fig, ax = plt.subplots()
				decompose_long.seasonal.plot()
				st.pyplot(fig)
				
				st.subheader('residual')
				fig, ax = plt.subplots()
				decompose_long.resid.plot()
				st.pyplot(fig)
		
		if st.button('acf & pacf'):
			with st.beta_container():

				st.subheader('Fatalities acf')
				fig, ax = plt.subplots()
				sm.tsa.graphics.plot_acf(dat_2020.fatalities, lags=20, ax=ax)
				st.pyplot(fig)

				st.subheader('Fatalities pacf')
				fig, ax = plt.subplots()
				sm.tsa.graphics.plot_pacf(dat_2020.fatalities, lags=20, ax=ax)
				st.pyplot(fig)

				st.subheader('Longitude acf')
				fig, ax = plt.subplots()
				sm.tsa.graphics.plot_acf(dat_2020.longitude, lags=20, ax=ax)
				st.pyplot(fig)

				st.subheader('Longitude pacf')
				fig, ax = plt.subplots()
				sm.tsa.graphics.plot_pacf(dat_2020.longitude, lags=20, ax=ax)
				st.pyplot(fig)

st.sidebar.header('Modelling and prediction')
if st.sidebar.checkbox("Modelling and prediction"):
	st.header('Modelling and prediction')
	data = clean_prepare()

	if st.sidebar.checkbox('Text classification'):
		st.subheader('What event is it?')

		with st.beta_expander('Learn more'):
			st.markdown(
				"""
			Event are encoded like this:

			1. 0 --> Battles.
			2. 1 --> Explosions/Remote violence.
			3. 2 --> Protests.
			4. 3 --> Riots.
			5. 4 --> Strategic developments.
			6. 5 --> Violence against civilians.
				""")


		file = open('/home/massock/Documents/Web App python/save_model/cmrConflictTextClassifier.pkl', 'rb')
		classifier = pickle.load(file)# classifier

		note = []
		nb_text = st.number_input('Give number of text.',1)

		for i in range(int(nb_text)):
			
			note.append(st.text_area('Give source text only one:', key=i)) # note write by source media.
		
		note = np.array(note)

		
		#raw_file = st.text_input('Give csv file text path:')#load csv file
		st.write(note)

		label = ['Battles', 'Explosions/Remote violence', 'Protests', 'Riots', 'Strategic developments', 'Violence against civilians']

		if st.button('predict'):

			#ext = os.path.splitext(raw_file)[1]
			if nb_text > 1:

				proba = classifier.predict_proba(note)
				textp = ['probability of text'+str(i) for i in range(int(nb_text))]
				prob = pd.DataFrame(proba, columns=label, index=textp)
				st.success('Prediction is ok, see probability.')
				st.dataframe(prob)

			else:

				pred = classifier.predict(note)
				proba = classifier.predict_proba(note)
				res = f'Event is {label[pred[0]]} with the probability of {100*proba[0][pred][0]}%'

				prob = pd.DataFrame(100*proba[0], columns=['probability(%)'], index=label)
				st.success(res)
				st.dataframe(prob)

	if st.sidebar.checkbox('Var Model'):
		st.subheader('VAR Model: Evaluation phase')
		with st.beta_expander('Learn more'):
			st.markdown(
			"""
			VAR model means Vector Autoregressive model. We use this model when our data 
			have multivariables .

			**N.B**: we use only 2 years  data 2019 and 2020. 
			""")

		st.write('Preparing data for VAR model')
		with st.beta_expander('Learn more'):
			lesson = """
			The two important feature for our VAR model are longitude and fatalities. we are transformed it 
			using pandas pivot table to create two sub data longitude and fatalities with admin1 columns taken
			as feature.

			So, we merge this two sub data to have only one entire data. After, we use interpolation on missing value
			and make variation inflation factor to check which feature are given more multicolinearity (remove feature 
			giving more multicolinearity).  
			"""
			st.markdown(lesson)

		#function created
		@st.cache
		def prep_data_var():

			#longitude & fatalities
			region_fatalities = pd.pivot_table(data[data.year.isin([2019, 2020])], values='fatalities', index='event_date',
                                   columns='admin1', aggfunc=np.sum).fillna(0)
			region_longitude = pd.pivot_table(data[data.year.isin([2019,2020])], values='longitude', index='event_date',
                                  columns='admin1')

			region_longitude.interpolate(inplace=True, limit_direction='both')

			df1 = region_fatalities
			df2 = region_longitude

			region = df2.join(df1, lsuffix="_long", rsuffix="_fatal")

			#date periods
			region.index = pd.to_datetime(region.index)
			region = region.asfreq('D', how='start')
			region.interpolate(inplace=True, limit_direction='both')

			#vif 
			df = region.drop(columns=['Nord_fatal','Sud_fatal'])
			vif = np.linalg.inv(df.corr().to_numpy()).diagonal()
			vifs = pd.Series(vif, index=df.columns, name='VIF').sort_values(ascending=False)

			data_vif = region.drop(columns=['Nord_fatal','Sud_fatal','Middle Africa_long','Nord_long','Sud_long',
                               'Est_long','Adamaoua_long','Ouest_long'])

			return data_vif

		@st.cache	
		def cointegration_test(data):    
		    #checking stationarity
		    from statsmodels.tsa.vector_ar.vecm import coint_johansen

		    # if all absolute eigen values are less than 1 data are stationary
		    res = coint_johansen(data, -1, 1).eig

		    return res

		#show data 
		if st.checkbox('show/hide data.'):
			
			df = prep_data_var()
			st.dataframe(df)
			cols = df.columns #take columns

			#display data
			with st.beta_expander('Learn more'):
				st.markdown("""
				Centre_long means longitude for centre same for others.
				Centre_fatal means fatalities for centre same for others.
					""")
			#menu
			with st.beta_container():

				admin = st.multiselect('Select one or more administrative:', tuple(cols))
				#st.write(admin)
			#visualize
			if st.button('visualize'):

				if admin == []:
					st.error('No administrative region for plotting, please select one or more.')

				else:
					fig, ax = plt.subplots()
					df[admin].plot(subplots=True, ax=ax)
					st.pyplot(fig)

		if st.button('cointegration test'):
			with st.beta_expander('Learn more'):
				st.markdown("""
				Lütkepohl, H. 2005. New Introduction to Multiple Time Series
    			Analysis. Springer.
    				Cointegration test helps to find if data is stationary. if all eigenvalue of 
    			matrix cointegration is less than 1 then data is stationary.
				""")
			df = prep_data_var()
			st.write(cointegration_test(df))

		if st.checkbox('evaluate', key=1):
			df = prep_data_var()

			#split data
			tsa_train =df[df.index < '2020-6-01']
			tsa_test = df[df.index >= '2020-6-01']

			from statsmodels.tsa.api import VAR
			var_model = VAR(tsa_train)

			#find Lag order selection
			#p_order = var_model.select_order(15).selected_orders['aic']
			results = var_model.fit(maxlags=15, ic='aic')
			lag_order= results.k_ar

			#take lags
			n_lags = 10*lag_order
			nstep= 7

			#forecast
			forecast = results.forecast(tsa_train[-n_lags:].values, nstep)
			from sklearn.metrics import mean_squared_error
			hp =[]
			for i, u in enumerate(tsa_test.columns):
				hp.append(f'RMSE for {u} = {np.sqrt(mean_squared_error(tsa_test.values[-nstep:, i-1], forecast[:, i-1]))}.')

			st.write(hp)
			with st.beta_expander('Learn more'):
				st.markdown("""
					RMSE means Root Mean Squared Error. That return $x = x_{m} \pm rmse$.  Where $x_{m}$ is a estimated variable.
					""")
			nt = st.selectbox('select idx:', [i for i in range(16)])
			if st.button('instant Granger causality'):
				x = results.test_inst_causality(nt)
				st.write(f'{x.title}:\n\n{x.h0}\n\n{x.conclusion_str} {x.signif_str}.')
				st.dataframe(list(x.summary()))
				


		st.subheader('VAR Model: Forecasting phase')
		if st.checkbox('Forecasting'):
			from statsmodels.tsa.api import VAR

			data_vif = prep_data_var()
			all_model = VAR(data_vif, dates=data_vif.index, freq='D')

			res = all_model.fit(maxlags=5, ic='aic')# model
			lag_order = res.k_ar

			m_lags = st.slider('Select lags previous data:', 0, 30, 1)
			step = st.slider('Select steps for Forecasting:', 0, 15, 1)

			mean_for, lower_for, upper_for = res.forecast_interval(data_vif[-(lag_order + m_lags):].values, step)

			#dataframe
			mean = pd.DataFrame(mean_for, columns=data_vif.columns)
			lower = pd.DataFrame(lower_for, columns=data_vif.columns)
			upper = pd.DataFrame(upper_for, columns=data_vif.columns)

			#future dates
			future_date = pd.date_range(start=data_vif.index.max(), periods=step)
			mean.index = future_date
			lower.index = future_date
			upper.index = future_date

			name = st.multiselect('Select feature for plot:', list(mean.columns))

			#plotting
			with st.beta_container():

				if st.button('forecast'):
					for u in name:
						fig, ay = plt.subplots()
						ax = data_vif[-(30):][u].plot(label='Previous', ax=ay)
						mean[u].plot(ax=ax, label='Forecast')
						lower[u].plot(ax=ax, label='Lower')
						upper[u].plot(ax=ax, label='Upper')
						ay.legend(loc='best')
						ay.set_title(f'{u}')
						ay.fill_between(mean.index, lower[u], upper[u], color='gray', alpha=0.25)

						st.pyplot(fig)


			if st.checkbox('where the next conflict?'):
				st.markdown("""
				### Where and when will be the next conflict and how many fatalities will make it?
				""")
				with st.beta_expander('Learn more'):
					st.markdown("""
						In this section, the app predict where will be  the next conflict (if possible).
						The result shows that the only administrative region (latitude, longitude, fatalities) that app must study are:

						1. Cameroon 
						2. Centre
						3. Extreme-Nord
						4. Littoral
						5. Nord_Ouest
						6. Sud_Ouest
						""")

				step = st.slider('Select steps for Forecasting:', 0, 30, 1)

				future_date = pd.date_range(start=data_vif.index.max(), periods=step)
				forecast = pd.DataFrame(res.forecast(data_vif.values[-60:], step), columns=list(data_vif.columns), index=future_date)

				variable = st.multiselect('Select two names (vx_long, vx_fatal):', sorted(list(set(data_vif.columns)-set(['Adamaoua_fatal', 'Est_fatal',
								 'Middle Africa_fatal', 'Ouest_fatal']))))

				file = open('/home/massock/Documents/Web App python/save_model/cmrConflictRegression.pkl', 'rb')
				kneig = pickle.load(file)
				


				if st.button('location'):
					name = variable[0] # take longitude variable
					location = forecast[variable]
					geol = pd.DataFrame()

					if name[-4:] == 'long':
						st.success(f'R² for latitude is {97}%, RMSE for (longitude and fatalities) see evaluation phase.')

						location[name[:-4]+'lat'] = kneig.predict(location.iloc[:, 0].values.reshape(-1, 1))
						location[variable[1]] = location[variable[1]].astype('int64')

						#geol['lat'] = location.iloc[:, 2]
						#geol['lon'] = location.iloc[:, 0]

						st.dataframe(location.iloc[:, [0,2,1]])
						#st.map(geol)

					else:
						st.error('Please respect this order (vx_long, vx_fatal) to find location.')





st.sidebar.header('Contact')
if st.sidebar.button('contact me'):

	doc = """
	cmr conflict data can be download here [cmr conflict data](https://data.humdata.org/dataset/acled-data-for-cameroon)

	if you want to contact me take this email link: [lumierebatalong@gmail.com](lumierebatalong@gmail.com)

	Thank you to use this app.
	"""

	st.markdown(doc)
	st.balloons()









		









			
		
	



			

		

		
				















		






	










