def wikipedia_tool ():
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
    tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    print(tool.name)
    print(tool.description)
    print(tool.args)
    print(tool.return_direct)

    response = tool.run({"query": "Langchain"})
    print(response)

def custom_tool ():
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_core.pydantic_v1 import BaseModel, Field

    class WikiInputs(BaseModel):
        """Inputs to the wikipedia tool."""

        query: str = Field(
            description="query to look up in Wikipedia, should be 3 or less words"
        )
    
    api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
    tool = WikipediaQueryRun(
        name="wiki-tool",
        description="look up things in wikipedia",
        args_schema=WikiInputs,
        api_wrapper=api_wrapper,
        return_direct=True,
    )
    print(tool.name)
    print(tool.description)
    print(tool.args)
    print(tool.return_direct)
    response = tool.run("langchain")
    print(response)


# wikipedia_tool ()
custom_tool()