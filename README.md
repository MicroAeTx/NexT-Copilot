# NexT-Copilot
There are Chinese and English versions, and the English version is below

# Chinese

使用例如LLama，LMstudio，OpenAI的GPT，谷歌的Gemini，Midjourney，DELL3这些AI应用的API甚至是直接在此项目上部署AI模型创建一个可以像Microsoft Copilot一样运行的工具

日程：

①创建GUI并实现基于OpenVino的本地语言模型调用 ✅

②实现语言模型对工具工具的调用

③集成SDXL_TURBO大模型

④实现音频生成

⑤实现在Windows上对电脑的直接控制

⑥打包为EXE


# 运行方法

①

```python
git clone https://github.com/MicroAeTx/NexT-Copilot.git
```
②

```python
git clone https://www.modelscope.cn/fuzirui/MiniCPM-2B-OpenVino-int8.git
```

将上一步下载的文件夹里的所有文件放置到  NexT-Copilot/minicpm-2b-dpo/INT8_compressed_weights中

③

```python
cd NexT-Copilot

python -m venv .venv

cd .venv/Scripts

./Activate.ps1

python -m pip install -r requirements.txt
```

④

打开llm-chatbot.py，转到第二行，设置你的模型的绝对路径

```python
python llm-chatbot.py
```

如果一切正常的话，命令窗口中将会返还一个链接，在浏览器中打开它即可运行

注意⚠目前仅支持英特尔设备运行


# English

Utilize APIs from AI applications such as LLama, LMstudio, OpenAI’s GPT, Google’s Gemini, Midjourney, DELL3, or even deploy AI models directly on this project to create a tool that can run like Microsoft Copilot.

Agenda:

① Create GUI and implement local language model invocation based on OpenVino ✅

② Implement language model invocation for the tool

③ Integrate SDXL_TURBO large model

④ Implement audio generation

⑤ Implement direct control of the computer on Windows

⑥ Package as EXE

# How to Run

①

```python
git clone https://github.com/MicroAeTx/NexT-Copilot.git
```
②

```python
git clone https://www.modelscope.cn/fuzirui/MiniCPM-2B-OpenVino-int8.git
```
Place all files from the downloaded folder into NexT-Copilot/minicpm-2b-dpo/INT8_compressed_weights.

③

```python
cd NexT-Copilot

python -m venv .venv

cd .venv/Scripts

./Activate.ps1

python -m pip install -r requirements.txt
```
④

Open llm-chatbot.py, go to the second line, and set the absolute path of your model.

```python
python llm-chatbot.py
```

If everything goes well, a link will be returned in the command window. Open it in your browser to run.

Note⚠ Currently only supports Intel devices.
