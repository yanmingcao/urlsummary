import sys
import requests
import textwrap
import re
import openai
from bs4 import BeautifulSoup
from langdetect import detect
from time import time, sleep

DEBUG = False

MAX_SUMMARY_LENGTH = 300
MAX_TOKENS = 400
MAX_CONTENT = 1800 

def get_content(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string
        content = soup.get_text()
    else:
        print("Error: Unable to get content from URL")
        exit(1)

    if DEBUG: # log to file
        filename = '%d_gpt3.txt' % int(time())
        with open('gpt3_logs/%s' % filename, 'w', encoding='utf-8') as outfile:
            outfile.write(title + '\n\n' + content + '\n\n')
            outfile.write('Content length: %d' % len(content))

    return title, content

def gpt3_completion(prompt, 
                    engine='text-davinci-003', 
                    max_tokens=MAX_TOKENS,
                    top_p=1,
                    temperature=0.5
                    ):    
    max_retry = 5
    retry = 0

    while True:
        try:
            response = openai.Completion.create( 
                prompt=prompt,
                engine=engine,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature
                )
            text = response.choices[0].text.strip()
            text = re.sub('\s+', ' ', text)

            if DEBUG:  # log to file
                filename = '%d_gpt3.txt' % int(time())
                with open('gpt3_logs/%s' % filename, 'w', encoding='utf-8') as outfile:
                    outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text + '\n\n')
                    outfile.write('Prompt length: %d\n\nResponse length: %d' % (len(prompt), len(text)))
            
            return text
        except Exception as oops:
            print("Error communicating with OpenAI:", oops)
            retry += 1
            if retry >= max_retry:
                print("Error: Unable to call OpenAI API")
                exit(1)
            sleep(1)


def summarize_chunk(text):
    prompt = "Summarize the following text in %d characters in Simplified Chinese:\n\n %s \n\n" % (MAX_SUMMARY_LENGTH, text)
    text = gpt3_completion(prompt)
    return text


def summarize_text(text, length):
    summary = text
    while (len(summary) > length):
        chunks = textwrap.wrap(summary, MAX_CONTENT)
        chunk_summaries = list()
        for chunk in chunks:
            chunk_summary = summarize_chunk(chunk)
            chunk_summaries.append(chunk_summary)

        summary = '\n\n'.join(chunk_summaries)

    return summary


def translate_text(text):
    prompt = "Translate the following text to Simplified Chinese:\n\n" + text + "\n\n"
    text = gpt3_completion(prompt)
    return text


if __name__ == '__main__':
    # Check if the correct number of arguments have been provided
    if len(sys.argv) < 2:
        print("Usage: python urlsummary.py <url> <summary_length>")
        sys.exit(1)

    url = sys.argv[1]
    if len(sys.argv) > 2:
        summary_length = int(sys.argv[2])
    else:
        summary_length = MAX_SUMMARY_LENGTH
    
    # load the API key
    with open('openaiapikey.txt', 'r', encoding='utf-8') as infile:
        openai.api_key = infile.read()
  
    # print("Retrieving content from URL...")
    title, content = get_content(url)
    language = detect(content)
    # print("title: %s\ncontent length:%d\nlanguage:%s\n" % (title, len(content), language))

    # print("Calling OpenAI to summarize the content...")
    summary = summarize_text(content, summary_length)

    if language != "zh-cn":
        title = translate_text(title)
    
    print("标题: ", title)
    print("链接: ", url)
    print("摘要: ", summary)
