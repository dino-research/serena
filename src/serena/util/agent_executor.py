import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    InvalidParamsError,
    SendStreamingMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
# from agno.agent import Agent
from agno.agent import (
    Agent,
    ToolCallStartedEvent,
    ToolCallCompletedEvent,
    RunResponseCompletedEvent
)

logger = logging.getLogger(__name__)

class GenericAgentExecutor(AgentExecutor):
    """AgentExecutor used by the tragel agents."""

    def __init__(self, agent: Agent):
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        logger.info(f'Executing agent {self.agent.agent_name}')
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()

        task = context.current_task

        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.contextId)
        
        async for event in self.agent.arun(query, stream=True, stream_intermediate_steps=True, session_id=task.contextId):
            if isinstance(event, ToolCallStartedEvent):
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        event.rool.tool_name,
                        task.contextId,
                        task.id
                    ),
                )
                continue
            if isinstance(event, ToolCallCompletedEvent):
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        event.rool.result,
                        task.contextId,
                        task.id
                    ),
                )
                continue
            if isinstance(event, RunResponseCompletedEvent):
                part = TextPart(text=event['content'])
                
                await updater.add_artifact(
                    [part],
                    name=f'{self.agent.name}-result',
                )
                await updater.complete()
                break
                

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
