from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.base_language import BaseLanguageModel

from models import Concept, ConceptRefinementState

# Instructions for concept refinement
concept_refinement_instructions = """You are an expert in explaining mathematical concepts clearly and concisely. 
1. First, review the topic {topic} and the description {description}. 
2. Examine the context and the desire voiced by the user to refine the explanation. The context is an interaction between the user and an AI assistant's attempt to generate a description that captures the topic and the user's description. 
{context}
3. Refine the description and the topic to make it clearer and upto the user's desire.
4. Your output should be a JSON object with the following schema: {{'concept': {{'topic': 'string', 'description': 'string'}}}} where topic is the name of the concept and description is the refined explanation of the concept.
"""

def refine_concept(state: ConceptRefinementState, model: BaseLanguageModel):
    """Refine the concept based on the current state."""
    response = None
    if len(state["messages"]) == 1:
        # Initial message, set up the question
        sys_message = SystemMessage(
            content="""You are an expert in explaining mathematical concepts. If user's request/question does not require some sort of mathematical descussion, you should respond with a very brief description of whichever concept you think is relevant. You shall always respond with json structured output with the following schema: {'concept': {'topic': 'string', 'description': 'string'}}.
Example Senario 1, user asks with little relevance for mathematical descussion:
User: Draw a triangle on the screen.
Assitant: {"concept": {"topic": "Triangle", "description": "A triangle is a polygon with three edges and three vertices. It is one of the basic shapes in geometry.""}}

Example Senario 2, user asks a question that requires a mathematical descussion:
User: What is the Pythagorean theorem?
Assitant: {"concept": {"topic": "Pythagorean theorem", "description": "The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse is equal to the sum of the squares of the lengths of the other two sides."}}"""
        )
        response = model.with_structured_output(Concept, method="json_mode").invoke(
            [sys_message, HumanMessage(content=state["messages"][0].content)]
        )
    else:
        # Subsequent messages, refine the concept
        sys_message = SystemMessage(
            content=concept_refinement_instructions.format(
                topic=state["concept"].topic,
                description=state["concept"].description,
                context="\n".join([msg.content for msg in state["messages"]]),
            )
        )

        response = model.with_structured_output(Concept, method="json_mode").invoke(
            [sys_message] + state["messages"]
        )
    print(response)
    return {
        "messages": [AIMessage(content=response["concept"]["description"])],
        "concept": Concept(
            topic=response["concept"]["topic"],
            description=response["concept"]["description"],
        ),
    }


def human_feedback(state: ConceptRefinementState):
    """Human feedback on the concept refinement."""
    user_feedback = input("User: ")
    return {
        "messages": [HumanMessage(content=user_feedback)],
    }


def should_proceed_to_manim(state: ConceptRefinementState, model: BaseLanguageModel):
    """Determine if the conversation should proceed to Manim code generation."""
    schema = {
        "should_proceed": "a boolean value indicating whether the conversation should proceed to the Manim code generation step."
    }
    sys_message = SystemMessage(
        content="""You are an expert in determining whether a given conversation between a user and an AI assistant should proceed to the Manim code generation step. Your output should be a JSON object with the following schema: {'should_proceed': boolean} where boolean is either true or false"""
    )
    response = model.with_structured_output(schema, method="json_mode").invoke(
        [sys_message] + state["messages"]
    )
    if response["should_proceed"]:
        print("----------------Proceeding to Manim code generation----------------")
        return "generate_manim_code"
    return "human_feedback"
