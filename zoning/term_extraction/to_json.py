import json
import re

import diskcache as dc
import openai
from openai import OpenAI

from tenacity import retry, retry_if_exception_type, wait_random_exponential

from ..utils import get_jinja_environment, get_project_root, cached


client = OpenAI()
template = get_jinja_environment().get_template("to_json.pmpt.tpl")
cache = dc.Cache(get_project_root() / ".diskcache")

def get_json(text):
    if text is None:
        return None
    #return json.loads(text)
    #match = re.search(r"```json\n(\{.*?\})\n```", text, re.DOTALL)
    #match = re.search(r"(\{.*?\})\n", text, re.DOTALL)
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    return json.loads(match.group(1)) if match else None

@cached(cache, lambda *args: json.dumps(args))
@retry(
    retry=retry_if_exception_type(
        (
            openai.APIError,
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.Timeout,
        )
    ),
    wait=wait_random_exponential(multiplier=1, max=60),
)
def to_json(x):
    # TODO: Is there a way to share this implementation with our generic prompt
    # function?
    resp = client.chat.completions.create(
        #model="gpt-4-turbo",
        model="gpt-4o",
        temperature=0.0,  # We want these responses to be deterministic
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": template.render(x=x)
            },
        ],
    )
    top_choice = resp.choices[0]
    text = top_choice.message.content
    return get_json(text)

