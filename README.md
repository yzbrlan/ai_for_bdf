## Project for work 
**Benchmarking Modern Large Language Models on MSC Nastran BDF input files: A Comprehensive Study**

The project is orgnised by features:
- LLM choice (an interface allowing user to choose different model from different source as remote API or local models hosted by Ollama)
- Custom *System prompt* definition
- Ability to do batch processing 
- Ability to output result in external text files

The post-processing of data is needed (comparison with plot)


## 运行环境
python 3.8

```shell script
pip install -r requirements.txt
```
recommand install in editable mode on the host machine
```shell script
pip install -e .
```
