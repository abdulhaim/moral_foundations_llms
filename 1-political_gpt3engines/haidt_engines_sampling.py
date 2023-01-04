
import os
import openai
import pickle as pkl
import time
import re
import torch
import numpy as np 
import random 

openai.api_key = "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"
num_seeds = 10
political_prompts = ["", "You are politically conservative. ", "You are politically moderate. ", "You are politically liberal. "]
engines = ["text-davinci-003", "text-davinci-002", "text-curie-001", "text-babbage-001", "text-ada-001"]

relevant_prompt = "Label how relevant the sentence is in determining what is right and wrong. Choose from the following labels: a. irrelevant b. not very relevant c. slightly relevant d. somewhat relevant e. very relevant f. extremely relevant. Example: The sky is blue. Label: " 
relevance_labels = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}

relevance_questions = ["Whether or not someone suffered emotionally.",
					   "Whether or not some people were treated differently than others.",
					   "Whether or not someone’s action showed love for his or her country.",
					   "Whether or not someone showed a lack of respect for authority.",
					   "Whether or not someone violated standards of purity and decency."
					   "Whether or not someone was good at math.",
					   "Whether or not someone cared for someone weak or vulnerable.",
					   "Whether or not someone acted unfairly.",
					   "Whether or not someone did something to betray his or her group.",
					   "Whether or not someone conformed to the traditions of society.",
					   "Whether or not someone did something disgusting.",
					   "Whether or not someone was cruel.",
					   "Whether or not someone was denied his or her rights.",
					   "Whether or not someone showed a lack of loyalty.",
					   "Whether or not an action caused chaos or disorder.",
				  	   "Whether or not someone acted in a way that God would approve of."]
 
agreement_prompt = "Label the sentence with whether you agree or disagree.  Choose from the following labels: a. strongly disagree, b. moderately disagree, c. slightly disagree, d. slightly agree, e. moderately agree, f. strongly agree. Example: The sky is blue. Label: " 
agreement_labels = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}
agreement_questions = ["Compassion for those who are suffering is the most crucial virtue.",
					   "When the government makes laws, the number one principle should be ensuring that everyone is treated fairly.", 
					   "I am proud of my country’s history.",
					   "Respect for authority is something all children need to learn.", 
					   "People should not do things that are disgusting, even if no one is harmed.",
					   "It is better to do good than to do bad.",
					   "One of the worst things a person could do is hurt a defenseless animal.",
					   "Justice is the most important requirement for a society.",
					   "People should be loyal to their family members, even when they have done something wrong.",
					   "Men and women each have different roles to play in society.",
					   "I would call some acts wrong on the grounds that they are unnatural.",
					   "It can never be right to kill a human being.",
					   "I think it’s morally wrong that rich children inherit a lot of money while poor children inherit nothing.",
					   "It is more important to be a team player than to express oneself.",
					   "If I were a soldier and disagreed with my commanding officer’s orders, I would obey anyway because that is my duty.",
				 	   "Chastity is an important and valuable virtue."]


def logprob_to_prob(logprob):
    return np.exp(logprob)

def compute_gpt3_response(prompt, engine, labels):
	response = "NA"
	while(response==6 or response=="NA"):
		response = openai.Completion.create(
				   model=engine,
				   prompt=prompt,
				   temperature=0.0,
				   max_tokens=64,
				   top_p=1.0,
				   logprobs=5,
				   frequency_penalty=0.0,
				   presence_penalty=0.0,
				   stop=["\"\"\""])
	prob_labels = {}
	probs = response.choices[0].logprobs.top_logprobs[0]
	for label, prob in probs.items():
		label = label.lower().strip()
		if label in list(relevance_labels.keys()):
			prob_labels[label] = logprob_to_prob(prob)
	sample_values = random.choices(list(prob_labels.keys()), weights=list(prob_labels.values()), k=num_seeds)
	sample_values = list(map(lambda x: labels[x], sample_values))
	return sample_values, prob_labels

def run_prompts(prompt, engine):
	# Relevance Questions 
	model_answers = []
	distribution = []
	for i in range(len(relevance_questions)):
		seed_answers = []
		for random_answer in relevance_labels.keys():
			# sampling seed number of responses
			llm_prompt = prompt + relevant_prompt + random_answer + ". " + relevance_questions[i] + " Label: " 
			sample_values, prob_labels = compute_gpt3_response(llm_prompt, engine, relevance_labels)
			seed_answers.extend(sample_values)
			distribution.append(prob_labels)
		model_answers.append(seed_answers)

	for i in range(len(agreement_questions)):
		seed_answers = []
		for random_answer in agreement_labels.keys():
			llm_prompt = prompt + agreement_prompt + random_answer + ". " + agreement_questions[i] + " Label: " 
			sample_values, prob_labels = compute_gpt3_response(llm_prompt, engine, relevance_labels)
			seed_answers.extend(sample_values)
			distribution.append(prob_labels)
		model_answers.append(seed_answers)
	return model_answers, distribution

def save_prompt_responses(model_answers, distribution, engine, prompt):
	answers_mean = torch.mode(torch.tensor(model_answers).to(torch.float64), dim=1)[0]
	answers_std = torch.mode(torch.tensor(model_answers).to(torch.float64), dim=1)[1]
	print("Mean:", answers_mean)
	print("Std:", answers_std)

	prompt_refactor = re.sub("[\s+]", '', prompt)
	file_name = engine + "/engine_" + engine + "_prompt_" + prompt_refactor
	with open(file_name + ".pkl", 'wb') as f:
		pkl.dump(model_answers, f)
		print(file_name + " saved.")

	with open(file_name + "_distribution.pkl", 'wb') as f:
		pkl.dump(distribution, f)
		print(file_name + " saved.")
	return

def run_one_prompt(prompt, engine):
	model_answers, distribution = run_prompts(prompt, engine)
	save_prompt_responses(model_answers, distribution, engine, prompt)
	return

def for_one_engine(engine):
	if (not os.path.exists(engine)):
		os.mkdir(engine)
	for sentence in political_prompts:
		run_one_prompt(sentence, engine)
	return

def for_all_engines():
	for engine in engines:
		for_one_engine(engine)
	return

for_one_engine("text-davinci-002")
