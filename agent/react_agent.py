from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (
    rag_summarize, 
    get_user_id, 
    get_current_time, 
    fetch_user_records, 
    fill_context_for_report
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    """
    工业软件平台多模式协同智能问答系统 - ReAct智能体
    
    核心功能：
    1. 工业技术文档智能问答
    2. 众包项目需求匹配
    3. 众创方案咨询
    4. 众扶政策解读
    5. 个性化报告生成
    
    工作流程：
    - 用户提问 -> ReAct思考循环 -> 工具调用 -> 信息整合 -> 专业回答
    - 支持流式输出，提升用户体验
    - 自动识别报告生成场景，切换专用提示词
    """
    
    def __init__(self):
        # 定义智能体可用工具列表
        tools = [
            rag_summarize,          # RAG检索工具：查询专业知识库
            get_user_id,            # 获取用户ID
            get_current_time,       # 获取当前时间
            fetch_user_records,     # 查询用户历史记录
            fill_context_for_report # 标记报告生成场景
        ]
        
        # 创建ReAct智能体
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=tools,
            middleware=[
                monitor_tool,           # 工具调用监控
                log_before_model,       # 模型调用前日志
                report_prompt_switch    # 动态提示词切换
            ]
        )

    def execute_stream(self, query: str):
        """
        流式执行智能体推理
        
        参数：
            query: 用户提问字符串
            
        返回：
            生成器，逐块返回回答内容
        """
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }

        # 初始化上下文，默认非报告场景
        context = {"report": False}
        
        # 流式处理
        for chunk in self.agent.stream(input_dict, stream_mode="values", context=context):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"

    def execute_sync(self, query: str) -> str:
        """
        同步执行智能体推理（非流式）
        
        参数：
            query: 用户提问字符串
            
        返回：
            完整的回答字符串
        """
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }
        
        context = {"report": False}
        result = self.agent.invoke(input_dict, context=context)
        
        # 提取最后一条消息的内容
        if result and "messages" in result:
            return result["messages"][-1].content
        
        return ""


if __name__ == '__main__':
    # 测试用例
    agent = ReactAgent()
    
    print("=" * 60)
    print("测试1：工业技术文档查询")
    print("=" * 60)
    test_query_1 = "智能制造中的工业互联网平台有哪些核心技术架构？"
    for chunk in agent.execute_stream(test_query_1):
        print(chunk, end="", flush=True)
    
    print("\n" + "=" * 60)
    print("测试2：众包项目咨询")
    print("=" * 60)
    test_query_2 = "如何参与制造业数字化转型的众包项目？"
    for chunk in agent.execute_stream(test_query_2):
        print(chunk, end="", flush=True)
    
    print("\n" + "=" * 60)
    print("测试3：生成个人使用报告")
    print("=" * 60)
    test_query_3 = "给我生成我的使用报告"
    for chunk in agent.execute_stream(test_query_3):
        print(chunk, end="", flush=True)
