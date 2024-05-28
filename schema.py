from langchain_core.pydantic_v1 import BaseModel, Field


class QuestionData(BaseModel):
    question: str

class QuestionAnswer(BaseModel):
    question: str = Field(description="question asked by user")
    answer: str = Field(description="answer from model")
