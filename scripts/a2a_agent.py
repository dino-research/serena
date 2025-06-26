# type: ignore

import json
import logging
import sys

from pathlib import Path

import click
import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import AgentCard

from agno.models.google.gemini import Gemini
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from sensai.util import logging
from sensai.util.helper import mark_used

from serena.agno import SerenaAgnoAgentProvider
from serena.util.agent_executor import GenericAgentExecutor

mark_used(Gemini, OpenAIChat)

logging.configure(level=logging.INFO)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10101)
def main(host, port):
    logging.info(f'Starting server on {host}:{port}')
    model = OpenAIChat(id="gpt-4o")
    
    with Path.open("agent_cards/code_review_agent.json") as file:
        data = json.load(file)
    agent_card = AgentCard(**data)
    client = httpx.AsyncClient()
    request_handler = DefaultRequestHandler(
        agent_executor=GenericAgentExecutor(agent=SerenaAgnoAgentProvider.get_agent(model)),
        task_store=InMemoryTaskStore(),
        push_notifier=InMemoryPushNotifier(client),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    logging.info(f'Starting server on {host}:{port}')

    uvicorn.run(server.build(), host=host, port=port)

if __name__ == '__main__':
    main()