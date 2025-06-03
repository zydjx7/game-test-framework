@echo off
REM 设置您的OpenAI API密钥
set LLM_API_KEY=your_openai_api_key_here

REM 运行LLM测试
python run_tests.py --feature llm_ammo_test.feature --target assaultcube

REM 暂停，以便查看输出
pause 