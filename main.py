import streamlit as st
from keybert import KeyBERT
from functools import reduce
from transformers import pipeline
import re
from langchain.agents import load_tools
from langchain.agents import initialize_agent
import os
from pathlib import Path
import base64
import cv2
import requests
from pathlib import Path
import base64
from openai import OpenAI
from matplotlib import pyplot as plt
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.utilities import GoogleSerperAPIWrapper
from langchain import tools
from langchain.tools import Tool
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import TextLoader
from io import StringIO

os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPER_API_KEY"] = "c6ed9e313a750454de6140b985d8938630e49984"

def generate_hashtags(docs, n_hash, tone, cont_type, num, low_up):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(docs, diversity=0.5, use_mmr=True, top_n=5)
    keywords_list = [i[0] for i in keywords]
    delim = ","
    keywords_str = reduce(lambda x, y: str(x) + delim + str(y), keywords_list)

    search = GoogleSerperAPIWrapper(k=5, type='news')
    tools = Tool(
        name="Intermediate Answer",
        func=search.run,
        description="""Give me top 5 current trending News on the given input"""
    )
    serp_res = tools.run(keywords_str)

    res = len(re.findall(r'\w+', docs))
    if res >= 100:
        with open("file.txt", "w") as f:
            f.write(docs)
        loader = TextLoader("file.txt")
        docs = loader.load()

        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
        chain = load_summarize_chain(llm, chain_type="stuff")
        summary_content = chain.run(docs)
    else:
        summary_content = docs

    chat = ChatOpenAI(model='gpt-4', temperature=0)
    prompt = PromptTemplate(
        template="""
            Generate {n_hash} hashtags on the given input parameters below:
            1)Summarised content for your reference:{docs}
            2)Latest Trends:{trends}
            3)Tone of Hashtags:{tone} ,Tune hashtags according to tone strictly.
            4)Max length of each hashtag:{max_length}
            5)Content Type:{cont_type}
            6)Extra Points to Consider:i)Does need numbers in hashtag:{num}
                                       ii)Hashtag should contain only {low_up} characters
            Note:
            Hashtags generation should focus on above parameters strictly.

            Output Format:
            hashtag1
            hashtag2
            etc.,
        """,
        input_variables=["n_hash", "docs", "trends", "tone", "max_length", "cont_type", "num", "low_up"]
    )

    prompt_format_values = prompt.format_prompt(n_hash=n_hash, docs=summary_content, trends=serp_res, tone=tone,
                                                max_length=10, cont_type=cont_type, num=num, low_up=low_up)
    response = chat(prompt_format_values.to_messages()).content
    return response.split('\n')


def main():
    option = st.sidebar.selectbox(":orange[CHOOSE THE FEATURE]", ["Text Based HASHTAG Generator", "Image Based HASHTAG Generator"])
    if option == 'Text Based HASHTAG Generator':
        st.title("Hashtag Generator App")
        # Input fields
        docs = st.text_area("Enter Text Content", height=200)
        n_hash = st.slider("Number of Hashtags", min_value=5, max_value=10, value=5)
        tone = st.selectbox("Tone of Hashtags", ["CREATIVE", "INSPIRING", "CASUAL", "FRIENDLY", "OPTIMISTIC", "WARM", "FORMAL", "SERIOUS"])
        cont_type = st.selectbox("Content Type", ["BUSSINESS", "LIFESTYLE", "FOOD", "TRAVEL", "POLITICS"])
        num = st.checkbox("Include Numbers in Hashtags")
        low_up = st.radio("Hashtag Case", ["LOWERCASE", "UPPERCASE"])

        if st.button("Generate Hashtags"):
            hashtags = generate_hashtags(docs, n_hash, tone, cont_type, num, low_up)
            st.markdown("Generated Hashtags:")
            for i, hashtag in enumerate(hashtags, start=1):
                st.write(f"{i}. {hashtag}")
    elif option == 'Image Based HASHTAG Generator':
        st.title("Image Based HASHTAG Generation")

        # Upload image through Streamlit file uploader
        image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if image:
            # Convert the image to base64
            image_content = image.read()
            base64_image = base64.b64encode(image_content).decode("utf-8")

            # Display the image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Request to OpenAI API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
            }

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a system that always extracts information from an image in a text format"
                            }
                        ]
                    },

                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Interpret this Image and give me the relevant meta data
                                    This should only be subjective info about the image; no technical image
                                    specification is needed. Give the descriptive and contextual insights of the image
                                  ```"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "auto"
                                }
                            }
                        ]
                    }
                ],

                "max_tokens": 150
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]

            n_hash = st.slider("Number of Hashtags", min_value=5, max_value=10, value=5)
            tone = st.selectbox("Tone of Hashtags",
                                ["CREATIVE", "INSPIRING", "CASUAL", "FRIENDLY", "OPTIMISTIC", "WARM", "FORMAL",
                                 "SERIOUS"])
            cont_type = st.selectbox("Content Type", ["BUSSINESS", "LIFESTYLE", "FOOD", "TRAVEL", "POLITICS"])
            num = st.checkbox("Include Numbers in Hashtags")
            low_up = st.radio("Hashtag Case", ["LOWERCASE", "UPPERCASE"])

            if st.button("Generate Hashtags"):
                hashtags = generate_hashtags(content, n_hash, tone, cont_type, num, low_up)
                st.markdown("Generated Hashtags:")
                for i, hashtag in enumerate(hashtags, start=1):
                    st.write(f"{i}. {hashtag}")

if __name__ == "__main__":
    main()