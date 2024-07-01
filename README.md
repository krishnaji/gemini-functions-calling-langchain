# gemini-function-calling-langchain

Based on  https://python.langchain.com/v0.2/docs/templates/gemini-functions-agent/

## Installation

Install the LangChain CLI if you haven't yet

```bash
pip install -U langchain-cli
```

## Adding packages

```bash
langchain app add gemini-functions-agent
```


## Launch LangServe
```shell
export GPLACES_API_KEY=<your-api-key>
```

```bash
langchain serve
```
![](gemini-function-calling-langchain.gif)
