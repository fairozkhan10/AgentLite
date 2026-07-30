from pydantic import BaseModel


class AgentAct(BaseModel):
    """Using AgentAct class to design the agent self-actions and API-call actions

    :param name: action name
    :type name: str
    :param desc: the description/documents of this action
    :type desc: str, optional
    """

    name: str
    desc: str = None
    params: dict = None
    #: True when the LLM output could not be parsed into an action. The agent
    #: turns this into a corrective observation instead of executing anything,
    #: giving the model a chance to re-emit the step in the expected format.
    parse_failed: bool = False


ActObsChainType = list[tuple[AgentAct, str]]
