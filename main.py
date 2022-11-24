import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

header = st.container()
model_training = st.container()
results = st.container()





@st.cache(allow_output_mutation=True)
def get_data(filename):
	#load cleaned corpus data
	df=pd.read_csv("Data/clean_data.csv")

	#Load language model
	embedder = SentenceTransformer('all-MiniLM-L6-v2')

	# Embed all documents
	corpus_embeddings = embedder.encode(df['Sentence'], convert_to_tensor=True)
	return df,embedder,corpus_embeddings


def predict(query,embedder,corpus_embeddings):
	top_k = 5
	print(query)
	answer=[]
	query_embedding = embedder.encode(query, convert_to_tensor=True)

	# We use cosine-similarity and torch.topk to find the highest 5 scores
	cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
	top_results = torch.topk(cos_scores, k=top_k)


	posts_seen=set()
	n=0
	for score, idx in zip(top_results[0], top_results[1]):
		n+=1
		#Skip, if this post is already recommended
		post=df.iloc[idx.item()]['Post']
		# if post not in posts_seen:
		#   posts_seen.add(post)
		# print("\nPost: {}\nDocument: {}\nScore: {:.4f}".format(post,df.iloc[idx.item()]['Sentence'],score))
		answer.append({'link':df.iloc[idx.item()]['Link'],'sentence': df.iloc[idx.item()]['Sentence'],'score':score.item()})

	print(answer)
	pd.DataFrame(answer).to_csv("result.csv",index=False)

with header:
	st.title('Piazza Smart Search')
	st.text("We semantically search for student queries in previous semester's piazza posts")
	df,embedder,corpus_embeddings = get_data('Data/data.csv')


with model_training:

	input_text = st.text_input('Input Query')
	st.button("Search", on_click=predict, args=(input_text,embedder,corpus_embeddings), disabled=False)

with results:
	res=pd.read_csv("result.csv")
	
	st.write("\nTop 5 most similar posts are:")
	for idx in range(5):
		st.write(f"### #{idx+1} [Check Piazza]({res.iloc[idx]['link']})")
		st.write(f"**Text**: {res.iloc[idx]['sentence']}")
		st.write(f"**Score**:{res.iloc[idx]['score']}")
		
