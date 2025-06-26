from agno.models.anthropic.claude import Claude
from agno.models.google.gemini import Gemini
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.playground.playground import Playground
from agno.playground.serve import serve_playground_app
from sensai.util import logging
from sensai.util.helper import mark_used

from serena.agno import SerenaAgnoAgentProvider

mark_used(Gemini, Claude)

# initialize logging (Note: since this module is reimported by serve_playground_app and the logging configuration
# is extended by SerenaAgentProvider, we must handle this here conditionally)
if __name__ == "__main__":
    logging.configure(level=logging.INFO)
    model = OpenAIChat(id="gpt-4o")
    agent = SerenaAgnoAgentProvider.get_agent(model)
    response_stream = agent.run("Analyze source code in: /home/dino/Documents/novaguard-ai2", stream=True, stream_intermediate_steps=True)
    for event in response_stream:
        logging.info(f"TTTT: \n {event}")



# Define the model to use (see Agno documentation for supported models; these are just examples)
# model = Claude(id="claude-3-7-sonnet-20250219")
# model = Gemini(id="gemini-2.5-pro-exp-03-25")
# model = OpenAIChat(id="gpt-4o")
# model = Ollama(id="qwen2.5:7b")

# app = Playground(agents=[SerenaAgnoAgentProvider.get_agent(model)]).get_app()

# if __name__ == "__main__":
#     serve_playground_app("agno_agent:app", reload=False, log_config=None)
