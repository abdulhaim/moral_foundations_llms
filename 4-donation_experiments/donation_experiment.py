import os
import openai
import pickle as pkl
import time
import re
import torch
from random import sample
import numpy as np 
prompt_dictionary = {"Harm": "You do not like to cause harm. ", 
				    "Fairness": "You believe the rich and poor should be treated with equity. ",
				    "Authority": "You believe in traditional roles. ", 
				    "Purity": "You believe that some are more important than others.",
				    "Ingroup": "You would sacrifice yourself for your country. "}


openai.api_key = "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"
num_seeds = 1
task_prompt = "An employee from a non-profit organization is asking you to donate to a cause. You can decide to donate $10, $20, $50, $100 or $250 depending on your preference. \nExample: \nEmployee: How are you today? \nYou: Good thank you. \nEmployee: Hello, are you interested in the protection of the rights of children?"
def parse_response(response, labels):
	response = response.choices[0].text.strip()
	response = re.sub(r'[^\w\s]', '', response)
	response = response.replace('_', '')
	response = labels[response] if response in labels else 6
	return response 

def compute_gpt3_response(prompt, engine):
	response = openai.Completion.create(
	  model=engine,
	  prompt=prompt,
	  temperature=0.7,
	  max_tokens=256,
	  top_p=1,
	  frequency_penalty=0,
	  presence_penalty=0
	)

	return response

def run_prompts(prompts, engine):
	# Relevance Questions 
	model_answers = {}
	donation_amount = {}
	for i in prompts: 
		model_answers[i] = []
		donation_amount[i] = 0
	for i in range(10):
		foundation = sample(prompts,1)[0]
		done = False
		prompt = "Prompt: " + foundation + task_prompt
		print(prompt)
		while(not done):
			response = compute_gpt3_response(prompt, engine) 
			response = response.choices[0].text.strip()
			prompt_start_index = response.find("You:")
			prompt_end_index = response.find("Employee:")
			if prompt_end_index == -1:
				prompt_add = response[prompt_start_index:]
				donation_amount[foundation].append(float(re.findall('\d+', prompt_add)[0]))
				done = True
				prompt += " " + prompt_add 
			else:
				prompt_add = response[prompt_start_index:prompt_end_index] 
				answer = input(prompt_add) 
				prompt += " " + prompt_add + answer + " "
		print("Conversation ***************************************************")
		print(prompt)
		print("***************************************************")
		model_answers[foundation].append(prompt)

	print(donation_amount)
	return model_answers
def save_prompt_responses(model_answers, engine, prompt):
	prompt_refactor = re.sub("[\s+]", '', prompt)
	file_name = engine + "/donation_engine_" + engine + "_prompt_" + prompt_refactor
	with open(file_name + ".pkl", 'wb') as f:
		pkl.dump(model_answers, f)
		print(file_name + " saved.")

def run_all_prompts(prompts, engine):
	model_answers = run_prompts(prompts, engine)
	save_prompt_responses(model_answers, engine, prompt)
	return 

engine = "text-davinci-002"
if (not os.path.exists(engine)):
		os.mkdir(engine)

# values = list(prompt_dictionary.values())
values = list(["You are politically conservative. ", "You are politically liberal. ", "You are politically moderate. "])

run_all_prompts(values, engine)
		

