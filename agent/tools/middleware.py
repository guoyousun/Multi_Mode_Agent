from typing import Callable
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from utils.prompt_loader import load_system_prompts, load_report_prompts

from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage]
) -> ToolMessage:
    """
    工具调用监控中间件
    记录所有工具调用的详细信息，便于调试和审计
    """
    tool_name = request.tool_call['name']
    tool_args = request.tool_call['args']
    
    logger.info(f"[tool_monitor] 执行工具：{tool_name}")
    logger.info(f"[tool_monitor] 传入参数：{tool_args}")
    
    try:
        result = handler(request)
        logger.info(f"[tool_monitor] 工具 {tool_name} 调用成功")
        
        # 特殊处理：报告生成场景标记
        if tool_name == "fill_context_for_report":
            request.runtime.context["report"] = True
            logger.info("[tool_monitor] 已标记为报告生成场景")

        return result
    except Exception as e:
        logger.error(f"[tool_monitor] 工具 {tool_name} 调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    """
    模型调用前日志中间件
    记录即将调用模型时的状态信息
    """
    message_count = len(state['messages'])
    latest_message = state['messages'][-1] if message_count > 0 else None
    
    logger.info(f"[log_before_model] 即将调用模型，当前消息数：{message_count}")
    
    if latest_message:
        msg_type = type(latest_message).__name__
        content_preview = str(latest_message.content)[:200] if latest_message.content else "(空)"
        logger.debug(f"[log_before_model] 最新消息类型：{msg_type} | 内容预览：{content_preview}")
    
    # 检查是否为报告生成场景
    is_report = runtime.context.get("report", False)
    if is_report:
        logger.info("[log_before_model] 当前为报告生成场景，将使用报告专用提示词")
    
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    """
    动态提示词切换中间件
    根据上下文判断是否使用报告专用的提示词模板
    """
    is_report = request.runtime.context.get("report", False)
    
    if is_report:
        logger.info("[report_prompt_switch] 切换到报告生成专用提示词")
        return load_report_prompts()
    else:
        logger.debug("[report_prompt_switch] 使用标准系统提示词")
        return load_system_prompts()