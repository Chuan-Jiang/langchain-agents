#from cgi import print_exception
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable



MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

#--- Tools   
@tool
def get_product_price(product:str) -> float:
    '''Look up the price of a product in the catalog'''
    print(f" >> executing get_product_price(product = '{product}')")
    price = {'laptop' :12222, 'headphone' : 11, 'keyboard' : 1}
    return price.get(product, 0)

@tool
def  apply_discount(price:float, discount_tier:str) -> float :
    '''apply a discount tier to a price and return the final price.
    Avaliable tiers: bronze, silvler and gold.'''

    print(f' >>> executing apply_discount(price = {price}, discount_tier = {discount_tier})')
    discount_percentages = {"bronze":5,"silver":12, "gold":25}
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1 - discount/100),2)


@traceable(name = "gugugaga")
def run_agent(question: str):
    tools = [get_product_price,apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f'ollama:{MODEL}', temperature = 0 )
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content = (
                """
                you are a helpfull shopping assitant
                you have access to a product catalog tool and a discount tool
                rules:
                1, never guess price. you can get price by using get_product_price()
                2, only calculate discount after have the real price 
                3, always use the provied tools.

                """
            )
        ),
        HumanMessage(content = question)
    ]

    #### loop
    for iteration in range(1,MAX_ITERATIONS + 1):
        print(f"\n ---iteration {iteration} -----")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        print(f'ai returen content: {ai_message.content}')


        if not tool_calls:
            print(f'\n Final Anser: {ai_message.content}')
            return ai_message.content 

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get('args')
        tool_call_id = tool_call.get('id')

        print(f'Tool:{tool_name} | args:{tool_args}')

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f'Tool {tool_name} not found')

        observation = tool_to_use.invoke(tool_args)
        print(f'tools result {observation}')


        messages.append(ai_message)
        messages.append(
            ToolMessage(content = str(observation), tool_call_id = tool_call_id)
        )



if __name__ == "__main__":
    print("Hello langchain agent (.bind_tools)!")
    print()
    #print(get_product_price("laptop"))
    result = run_agent("What is the price of a laptop after applysing a gold discount?")

