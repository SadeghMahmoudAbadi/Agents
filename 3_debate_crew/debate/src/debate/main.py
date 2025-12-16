import sys
import warnings

from debate.crew import Debate

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """ Run the crew. """
    
    inputs = {
        'motion': 'Countries should accept cryptocurrencies as legal tender.',
    }
    
    try:
        result = Debate().crew().kickoff(inputs=inputs)
        print(result.raw)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
