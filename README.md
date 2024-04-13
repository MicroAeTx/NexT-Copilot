# NexT-Copilot
There are Chinese and English versions, and the English version is below

#Chinese

使用例如LLama，LMstudio，OpenAI的GPT，谷歌的Gemini，Midjourney，DELL3这些AI应用的API甚至是直接在此项目上部署AI模型创建一个可以像Microsoft Copilot一样运行的工具

日程：

①创建GUI并实现基于OpenVino的本地语言模型调用 ✅

②实现语言模型对工具工具的调用

③集成SDXL_TURBO大模型

④实现音频生成

⑤实现在Windows上对电脑的直接控制

⑥打包为EXE


#运行方法

①
  
```python

git clone https://www.modelscope.cn/fuzirui/MiniCPM-2B-OpenVino-int8.git

git clone https://github.com/MicroAeTx/NexT-Copilot.git

cd NexT-Copilot

python -m venv .venv

cd .venv/Scripts

./Activate.ps1

python -m pip install requirements.txt
```
②
将模型文件夹和llm-chatbot.py放在同一文件夹下

打开llm-chatbot.py，转到第二行，设置你的模型的绝对路径

```python
python llm-chatbot.py
```

如果一切正常的话，命令窗口中将会返还一个链接，在浏览器中打开它即可运行

注意⚠目前仅支持英特尔设备运行




#English

Using APIs for AI applications such as LLama, LMstudio, OpenAI's GPT, Google's Gemini, Midjourney, DELL3, or even deploying AI models directly on this project, create a tool that can run like Microsoft Copilot

Schedule:

(1) Create a GUI and implement local language model calls  based on OpenVino ✅

(2) Implement the language model's invocation of tools

(3) Integrate SDXL_TURBO large models

(4) Realize audio generation

(5) Realize direct control of the computer on Windows

(6) Package as EXE


#Run method

(1)
  
```python

git clone https://www.modelscope.cn/fuzirui/MiniCPM-2B-OpenVino-int8.git

git clone https://github.com/MicroAeTx/NexT-Copilot.git

cd NexT-Copilot

python -m venv .venv

cd .venv/Scripts

./Activate.ps1

python -m pip install requirements.txt
```
(2)
Place the model folder and llm-chatbot.py under the same folder

Open the llm-chatbot.py, go to the second row, and set the absolute path of your model

```python
python llm-chatbot.py
```

If everything is fine, a link will be returned to the command window, which can be opened in the browser and will be ready to run

Note: ⚠ Only Intel devices are supported at this time
