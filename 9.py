from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pydantic import BaseModel, Field
class InstitutionInfo(BaseModel):
    founder: str
    founded_year: str
    branches: str
    employees: str
    summary: str
institution = input("Enter Institution Name: ")
wiki = WikipediaAPIWrapper()
wiki_data = wiki.run(institution)
template = """
Founder:
Founded Year:
Branches:
Employees:
Summary:
Text:
{text}
"""
prompt = PromptTemplate(
    input_variables=["text"],
    template=template
)
formatted_prompt = prompt.format(text=wiki_data)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
inputs = tokenizer(formatted_prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"])
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
