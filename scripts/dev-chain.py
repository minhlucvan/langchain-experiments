from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.utilities import GoogleSerperAPIWrapper, BashProcess
from langchain.chains import LLMChain
from langchain.chat_models.openai import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import sys
from io import StringIO

class PythonEditor:
    """Simulates a standalone PythonEditor."""

    def __init__(self):
        pass        

    def run(self, command: str) -> str:
        """Run command and returns anything printed."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        code = command
        match = re.search(r'```(.+?)```', code, re.DOTALL)
        if match is not None:
            code = match.group(1).strip('`')
        try:
            exec(code, globals())
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            output = str(e)
        return output

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.25,openai_api_key="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")
searh = GoogleSerperAPIWrapper(serper_api_key="266756080b5e71c313d442bf625aa7d41e8f233b")
bash = BashProcess()
python_editor = PythonEditor()

tools = [
    Tool(
        "PythonEditor",
        python_editor.run,
        """Usefull when you need to execute python code. Input should be a valid python script).
        If you expect output it should be printed out.""",
    ),
    Tool(
        "Bash",
        bash.run,
        """Usefull when you need to execute command prompt""",
    )
]

template = """Complete the following task as best you can, You have access to the following tools:

{tools}

Use the following format:

Task: the input task you must complete
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final output
Final output: the final output to the original task

Begin! Remember to be act as a Virtual Assistant, get information about your current running enviroiment, os, directory, etc. you only have access to the authorized tools.

Task: {input}
Thought: I need to develop a task list to accomplish this original task
{agent_scratchpad}"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final output:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final output:")[-1].strip()},
                log=llm_output,
            )
        # Splitting the text into action and input using regex
        action, action_input = (re.split(r'Action Input:', llm_output) + [''])[:2]

        # Removing "Action:" and "Action Input:" from the output
        action = action.strip().replace('Action: ', '')
        action_input = action_input.strip().replace('Action Input: ', '')
        action = action.strip().strip('`')
        action_input = action_input.strip().strip('`')

        if "PythonEditor" in action:
            action =  'PythonEditor'
        elif "Bash" in action:
            action = 'Bash'
        
        if not action or not action_input:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)

output_parser = CustomOutputParser()

llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    max_iterations=None,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

agent_executor.run("Find my current physical location")