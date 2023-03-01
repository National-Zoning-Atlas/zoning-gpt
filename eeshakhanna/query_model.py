import sys
import math
import pandas as pd
import openai

#TODO: Figure out where to store intermediate data...
#TODO: Figure out how to load the chunked data in here (as df or dataset?)

# login to openai using orgID and API key
openai.organization = sys.argv[1]
openai.api_key = sys.argv[2]


def token_price_estimator(prompt, model_type='davinci'):
  char_count = len(prompt)
  token_count = math.ceil(char_count / 4)
  if model_type == 'davinci':
    return token_count * 0.02 / 1000
  elif model_type == 'curie':
    return token_count * 0.002 / 1000
  elif model_type == 'babbage':
    return token_count * 0.0005 / 1000
  elif model_type == 'ada':
    return token_count * 0.0004 / 1000
  else:
    return 'invalid model type'


def gpt3_query_one_chunk(prompt_with_chunk, model_type="text-davinci-003"):
  # query gpt-3 model using prompt
  response = openai.Completion.create(
    model=model_type,
    prompt=prompt_with_chunk,
    temperature=0,
    max_tokens=100,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    logprobs = 1
    )
  # return the result and it's probability
  response_text = response["choices"][0]["text"]
  response_prob_dict = response["choices"][0]["logprobs"]["top_logprobs"]

  token_text = ""
  sum_prob = 0
  for x in response_prob_dict:
    token_text += list(x.keys())[0]
    sum_prob += list(x.values())[0]
    if token_text == response_text:
      break
  token_text = token_text.replace("\n", "")
  return token_text, math.exp(sum_prob)


def gpt3_query_all_chunks(query, chunks, model_type="text-davinci-003"):
  responses = []
  for i in range(len(chunks)):
    prompt = query + " " + chunks[i]
    text, prob = gpt3_query_one_chunk(prompt, model_type)
    responses.append((text, prob))
  return responses


def get_answer_all_chunks(responses):
  highest_prob = 0
  answer = ''
  index = 0
  for i in range(len(responses)):
    x = responses[i]
    if x[0] == 'NA':
      continue
    if x[1] > highest_prob:
      answer = x[0]
      highest_prob = x[1]
      index = i
  
  return answer, highest_prob, index