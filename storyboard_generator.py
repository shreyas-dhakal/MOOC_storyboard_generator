import os
import getpass
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import pandas as pd
from tabulate import tabulate
import openai
import warnings
warnings.filterwarnings("ignore")

#Validate OpenAI API key
def validate_openai_key(api_key):
    try:
        openai.api_key = api_key
        openai.models.list()
        return True
    except:
        return False

while True:
    api_key = getpass.getpass("Enter your OpenAI API Key: ")
    if validate_openai_key(api_key):
        os.environ["OPENAI_API_KEY"] = api_key
        print("API key is valid and has been set.")
        break
    else:
        print("Invalid API key. Please try again.")

#Variables for chunk sizes and chunk overlap for splitting the documents
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50

#Loading the models for llm, parsing and embedding
llm = ChatOpenAI(model="gpt-4o")
parse_llm = ChatOpenAI(model='o1')
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def document_parser(loader):
    print("parsing the document ...")
    pages = loader.load_and_split()
    vectorstore = InMemoryVectorStore.from_documents(
        pages,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever()
    text_question_gen = ''
    for page in pages:
        text_question_gen += page.page_content
    text_splitter_presentation = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)
    text_chunk_split = text_splitter_presentation.split_text(text_question_gen)
    docs_presentation = [Document(page_content = t) for t in text_chunk_split]
    return docs_presentation

def generate_presentation(docs_presentation):
    print("generating slides ...")
    prompt_template_question = """
    You are an expert in creating presentation slides for MOOC based on context provided.
    Your goal is to extract key-points, explanations and syntax for the presentation:

    -----------
    {text}
    -----------

    Create the content for the presentation. Make sure not to lose any important information.

    Points:
    """
    prompt_presentation = PromptTemplate(template = prompt_template_question, input_variables = ['text'])
    refine_template_presentation = """
    You are an expert in creating presentation slides for MOOC storyboard in the following format:
    Slide Content
    Slide 1: Title Slide
    (Course Title)

    Slide 2: Outline 
    (Topics Covered)

    Slide 3: Key Concept 1 
    (Definition): AI

    Your goal is to prepare presentation slide points(optimal number of slides, at least 5).
    Make sure the course outline matches the content which is presented later on.
    Refine the contents from the given text: {existing_answer}.
    INCLUDE the explanations and syntax in the slide content to make it informative.
    DO NOT generate any other text than the said task.

    ------------
    {text}
    -----------

    Given the new context, refine the points.
    If the context is not helpful, please create content based on the prepared points. 
    """

    refine_prompt_presentations = PromptTemplate(
        input_variables=['existing_answer', 'text'],
        template=refine_template_presentation,
    )

    #Use 'refine' method to generate and improve key points for the presentation
    presentation_gen_chain = load_summarize_chain(llm = llm, chain_type = 'refine', verbose = False, question_prompt = prompt_presentation, refine_prompt = refine_prompt_presentations)
    slide_content = presentation_gen_chain.invoke(docs_presentation)
    return slide_content

def generate_dialogues(slide_content):
    #Define Pydantic Models
    print("generating dialogues ...")
    class Presentation(BaseModel):
        slides: list = Field(description="All the contents of slides with the following keys only: slideNumber and content. NO SUBKEYS")
        dialogues: list = Field(description="All the lecturer dialogues with the following keys only: slideNumber and dialogue")

    parser = PydanticOutputParser(pydantic_object = Presentation)
    format_instruction = parser.get_format_instructions()

    template_string = """Your task is to generate the lecturer dialogues explaining all the terms and concepts in the slides.
    Use a elegant and pragmatic approach to make the dialogues complementing the slides. Add some extra information relevant to the content as well and make sure enough dialogue is there to present the slides propoerly.
    Seperate the slides and lecturer dialogues.

    {text}

    Maintain the line breaks and formats.
    Convert it into the given unstructured Pydantic Format.

    {format_instruction}
    """
    prompt = ChatPromptTemplate.from_template(template = template_string)
    message = prompt.format_messages(text = slide_content['output_text'], format_instruction = format_instruction)
    output = parse_llm.invoke(message)
    the_presentation = parser.parse(output.content)
    return the_presentation

def convert_to_table(the_presentation):

    slides = the_presentation.slides
    dialogues = the_presentation.dialogues

    combined = []
    #Join Slides and Dialogues according to their slide number
    for slide in slides:
        slide_num = slide['slideNumber']
        matching_dialogue = next((d for d in dialogues if d['slideNumber'] == slide_num), {})
        combined.append({
            'Slide Content': slide['content'],
            'Dialogue': matching_dialogue.get('dialogue', '')
        })
    df = pd.DataFrame(combined)
    return df

#Load the manual
loader = PyPDFLoader("manual.pdf")
docs = document_parser(loader)
slides = generate_presentation(docs)
presentation = generate_dialogues(slides)
df = convert_to_table(presentation)

#Save the storyboard as a csv file
df.to_csv("storyboard.csv", index = False)
print("saved the storyboard as storyboard.csv")


