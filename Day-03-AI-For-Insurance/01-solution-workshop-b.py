import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Step 1: Load environment variables from .env file
load_dotenv(".env")

# Step 2: Retrieve Azure OpenAI environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

model = AzureChatOpenAI(
    azure_deployment= "gpt-4o-mini",  # Your Azure deployment
    api_version     = OPENAI_API_VERSION,  # Your API version
    api_key         = AZURE_OPENAI_API_KEY,
    azure_endpoint  = AZURE_OPENAI_ENDPOINT,
    temperature     = 0,
    max_tokens      = None,
    timeout         = None,
)

# Output parser for the response
output_parser = StrOutputParser()

# Write the prompt template
template = """
You are an assistant helping a customer understand insurance policies. 
Please provide a detailed explanation of the {policy_type} insurance policy in {country}, focusing on the following aspects:
- **Coverage**: What does the insurance cover? Please list key treatments, services, or damages that are included in the policy.
- **Eligibility**: Who is eligible for this insurance? Include any age, residency, or medical conditions that may affect eligibility.
- **Exclusions**: Are there any exclusions or conditions not covered by this insurance policy?
- **Benefits**: What are the key benefits of this policy? What makes it stand out compared to other similar policies?
- **Premium Info**: What is the cost of the insurance policy? Please mention the payment options and any discounts available.
- **Claims Process**: How does a customer file a claim for this insurance? What steps are involved?

Be sure to provide clear, concise, and helpful information for the customer to easily understand. Your response should help the customer make an informed decision about the insurance policy.
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["policy_type", "country"]
)

# Set up the chain
chain = prompt | model | output_parser

# Invoke the chain with specific queries
response_health_thailand = chain.invoke({
    "policy_type": "health insurance",
    "country": "Thailand"
})
print(response_health_thailand)