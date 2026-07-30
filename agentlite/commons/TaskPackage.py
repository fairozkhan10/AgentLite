import time
import uuid

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class TaskPackage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    instruction: str
    completion: str = "active"
    # Callers throughout the codebase, the README and the tutorials pass
    # `task_creator=` / `task_executor=`. Accept both spellings so those keep
    # working instead of being silently dropped as unknown kwargs.
    creator: str = Field(
        default="", validation_alias=AliasChoices("creator", "task_creator")
    )
    # default_factory, not a class-body call: a plain default is evaluated once
    # at import time and then shared by every instance.
    timestamp: str = Field(default_factory=lambda: str(time.time()))
    answer: str = ""
    executor: str = Field(
        default="", validation_alias=AliasChoices("executor", "task_executor")
    )
    priority: int = 5
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def __str__(self):
        return f"""Task ID: {self.task_id}\nInstruction: {self.instruction}\nTask Creator: {self.creator}\nTask Completion:{self.completion}\nAnswer: {self.answer}\nTask Executor: {self.executor}"""
