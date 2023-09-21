# coro_agent

An agent tool built on langchain, which prompts the user for information by asking questions and completing prompts through user responses in interactive scenarios.

```python
from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
import json,os
from datetime import datetime
import pytz
from coro_agent.coro_agent import initialize_coro_agent, CoroAgentSuspend

memory = ConversationBufferMemory(memory_key='chat_history')

os.environ['OPENAI_API_KEY'] = '****'

llm = OpenAI(temperature=0)

def queryTime(action_input):
    '''
    查询此时此刻时间
    '''
    now = datetime.now()
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = now.astimezone(beijing_tz)
    return beijing_time.strftime("%Y-%m-%d %H:%M:%S")

def askUser(action_input: str):
    '''
    当你无法确定可以调用哪个工具或者调用工具缺少必要参数的时候，你可以询问用户获取想要的答案，如果用户回答不满足，你可以重复提问，输入为:[你想了解的问题]
    '''
    return CoroAgentSuspend(output=action_input)

def searchBill(action_input: str):
    '''
    用于查询指定日期账单, 查询参数不可编造，如果无法从用户提问中得知，需要向用户询问，需要的参数为json格式:
    {{
        "type": "账单类型:日账单|月账单", 
        "startDate": "开始日期, 如果是日账单需要精确到日,eg:20230627, 如果是月账单只需要精确到月,eg:202306", 
        "endDate": "结束日期,格式和startDate一致"
    }}
    '''
    params = json.loads(action_input)
    return '查询到账单如下：账单类型:{}\t日期:{}-{}\t 消费200元' .format(
        params['type'],
        params['startDate'],
        params['endDate'],    
    )

tools = [
    StructuredTool.from_function(searchBill),
    StructuredTool.from_function(askUser),
    StructuredTool.from_function(queryTime),
]


#agent_chain = initialize_coro_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
# The agent `CONVERSATIONAL_REACT_DESCRIPTION`` have a aalignant bug， see: https://github.com/langchain-ai/langchain/issues/10311

agent_chain = initialize_coro_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)

r = agent_chain.run(input='我想查账单')
print(f'>>>>>>>>{r}')
```
```test
> Entering new CoroAgentExecutor chain...
 我需要知道账单的类型，开始日期和结束日期
Action: askUser
Action Input: 请问您想查询的账单类型，开始日期和结束日期是多少？
Observation: output='请问您想查询的账单类型，开始日期和结束日期是多少？'
Thought:
> Finished chain.
>>>>>>>>请问您想查询的账单类型，开始日期和结束日期是多少？
```
```python
r = agent_chain.run(input='查询上个月的账单')
print(f'>>>>>>>>{r}')
```
```test
> Entering new CoroAgentExecutor chain...
{'agent_inputs': {'input': '我想查账单', 'chat_history': ''}, 'intermediate_steps': [], 'last_agent_action': AgentAction(tool='askUser', tool_input='请问您想查询的账单类型，开始日期和结束日期是多少？', log=' 我需要知道账单的类型，开始日期和结束日期\nAction: askUser\nAction Input: 请问您想查询的账单类型，开始日期和结束日期是多少？')}
 我现在知道了账单的类型，开始日期和结束日期
Action: searchBill
Action Input: {
    "type": "月账单", 
    "startDate": "202305", 
    "endDate": "202306"
}
Observation: 查询到账单如下：账单类型:月账单	日期:202305-202306	 消费200元
Thought: 我现在知道了账单的详细信息
Final Answer: 您查询的账单类型为月账单，日期为202305-202306，消费200元。

> Finished chain.
>>>>>>>>您查询的账单类型为月账单，日期为202305-202306，消费200元。
```