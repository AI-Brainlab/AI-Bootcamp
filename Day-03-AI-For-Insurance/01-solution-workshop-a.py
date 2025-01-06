### Example Solution:
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# Define the data structure for Personal Medical Information.
class MedicalRecord(BaseModel):
    patient_name: str = Field(description="name of the patient")
    date_of_birth: str = Field(description="date of birth of the patient")
    gender: str = Field(description="gender of the patient")
    medical_history: str = Field(description="medical history of the patient")
    medications: str = Field(description="current medications of the patient")
    allergies: str = Field(description="allergies the patient has")
    emergency_contact_name: str = Field(description="emergency contact's name")
    emergency_contact_phone: str = Field(description="emergency contact's phone number")
    primary_care_physician: str = Field(description="name of the primary care physician")

# Given query (patient medical document).
medical_document = """
Patient Name: John Doe
Date of Birth: 1985-06-15
Gender: Male
Medical History: John has a history of asthma and seasonal allergies. He underwent a knee surgery in 2018 due to an accident. No known chronic conditions or heart disease. 
Medications: Currently using Ventolin inhaler for asthma as needed. No regular medications.
Allergies: Allergic to penicillin.
Emergency Contact: Jane Doe (Wife), 555-1234
Primary Care Physician: Dr. Smith, General Practitioner, ABC Health Clinic.
"""

# Set up the JsonOutputParser and the prompt template.
parser = JsonOutputParser(pydantic_object=MedicalRecord)

prompt = PromptTemplate(
    template="Extract the necessary details from the medical record.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Assuming the model is defined in your pipeline
chain = prompt | model | parser
medical_info = chain.invoke({"query": medical_document})

# Display the extracted information
print(medical_info)