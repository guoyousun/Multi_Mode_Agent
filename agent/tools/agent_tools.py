import os.path
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path


rag = RagSummarizeService()


@tool(description="从向量库检索工业技术文档、众包项目信息、众创方案资料、众扶政策等专业内容")
def rag_summarize(query: str) -> str:
    """
    检索内容包括：
    - 工业技术文档：技术架构、功能模块、标准规范、实施方法、行业应用案例
    - 众包项目：项目需求、任务类型、参与流程、成功案例、风险防范
    - 众创方案：创新模式、技术支持、资源配置、合作机制、成果展示
    - 众扶政策：政策解读、扶持措施、申报条件、资助标准、申请流程
    
    参数：
        query: 检索关键词，应为简洁明确的查询语句
    
    返回：
        字符串类型的专业资料内容，包含与检索词匹配的精准解答和相关知识点
    """
    return rag.rag_summarize(query)


@tool(description="获取当前用户的唯一标识ID，用于查询个人相关的项目记录和使用报告")
def get_user_id() -> str:
    """
    获取当前发起请求的用户唯一标识(ID)，格式为数字字符串(如 "1001")
    
    使用场景：
    - 查询用户参与的众包项目
    - 生成个性化使用报告
    - 统计用户历史活动数据
    
    返回：
        字符串类型的用户ID (如 "1001", "1002" 等)
    """
    # TODO: 实际项目中应从认证系统或会话中获取真实用户ID
    # 此处为演示，返回模拟数据
    import random
    user_ids = ["1001", "1002", "1003", "1004", "1005"]
    return random.choice(user_ids)


@tool(description="获取系统当前时间信息，用于报告生成和时间相关的查询")
def get_current_time() -> str:
    """
    获取系统当前的日期和时间信息
    
    使用场景：
    - 生成带时间戳的报告
    - 查询特定时间段的项目信息
    - 统计数据的时间范围界定
    
    返回：
        字符串类型的时间信息，格式为 "YYYY-MM-DD HH:MM:SS"
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool(description="从外部数据库查询指定用户在指定时间段的使用记录和项目参与情况")
def fetch_user_records(user_id: str, time_range: str = "recent") -> str:
    """
    从外部系统中获取用户的历史使用记录和项目参与数据
    
    参数：
        user_id: 用户唯一标识ID
        time_range: 时间范围，可选值："recent"(最近一个月)、"quarter"(本季度)、"year"(本年度)、"all"(全部)
    
    返回：
        字符串类型的用户记录信息，包含：
        - 参与的众包项目列表
        - 提交的众创方案
        - 享受的众扶政策
        - 平台使用统计数据
        
        如果未检索到数据，返回空字符串
    """
    generate_external_data()
    
    try:
        if user_id not in external_data:
            logger.warning(f"[fetch_user_records] 未找到用户 {user_id} 的记录")
            return ""
        
        records = external_data[user_id]
        
        # 根据时间范围过滤数据
        if time_range == "recent":
            # 返回最近一个月的记录
            return str(list(records.values())[-1]) if records else ""
        elif time_range == "quarter":
            # 返回本季度的记录
            recent_items = list(records.items())[-3:]
            return str({k: v for k, v in recent_items})
        elif time_range == "year":
            # 返回本年度的记录
            return str(records)
        else:  # "all"
            return str(records)
            
    except Exception as e:
        logger.error(f"[fetch_user_records] 查询失败: {str(e)}")
        return ""


# 外部数据存储
external_data = {}

def generate_external_data():
    """
    从CSV文件加载外部数据到内存字典
    数据结构示例：
    {
        "1001": {
            "2025-01": {
                "参与项目": "智能制造优化方案众包",
                "提交方案": "生产线自动化改造建议",
                "享受政策": "技术创新扶持基金",
                "使用时长": "45小时",
                "贡献度": "高"
            },
            ...
        }
    }
    """
    if external_data:
        return
    
    external_data_path = get_abs_path(agent_conf.get("external_data_path", "data/external_data.csv"))
    
    if not os.path.exists(external_data_path):
        logger.warning(f"[generate_external_data] 外部数据文件 {external_data_path} 不存在，使用模拟数据")
        # 生成模拟数据
        _generate_mock_data()
        return

    try:
        with open(external_data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                _generate_mock_data()
                return
            
            # 跳过标题行
            for line in lines[1:]:
                arr = line.strip().split(",")
                if len(arr) < 6:
                    continue
                    
                user_id = arr[0].replace('"', "").strip()
                project = arr[1].replace('"', "").strip()
                solution = arr[2].replace('"', "").strip()
                policy = arr[3].replace('"', "").strip()
                usage_hours = arr[4].replace('"', "").strip()
                month = arr[5].replace('"', "").strip()

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][month] = {
                    "参与项目": project,
                    "提交方案": solution,
                    "享受政策": policy,
                    "使用时长": usage_hours,
                }
        
        logger.info(f"[generate_external_data] 成功加载 {len(external_data)} 个用户的外部数据")
        
    except Exception as e:
        logger.error(f"[generate_external_data] 加载外部数据失败: {str(e)}")
        _generate_mock_data()


def _generate_mock_data():
    """生成模拟的外部数据用于演示"""
    mock_projects = [
        "智能制造优化方案众包",
        "工业互联网平台建设",
        "数字化车间改造",
        "产品质量检测系统开发",
        "供应链协同平台设计"
    ]
    
    mock_solutions = [
        "生产线自动化改造建议",
        "数据采集与分析方案",
        "设备预测性维护策略",
        "工艺流程优化方案",
        "能源管理系统设计"
    ]
    
    mock_policies = [
        "技术创新扶持基金",
        "数字化转型补贴",
        "研发费用加计扣除",
        "人才引进政策支持",
        "知识产权保护资助"
    ]
    
    months = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
    user_ids = ["1001", "1002", "1003", "1004", "1005"]
    
    import random
    for user_id in user_ids:
        external_data[user_id] = {}
        for month in months:
            external_data[user_id][month] = {
                "参与项目": random.choice(mock_projects),
                "提交方案": random.choice(mock_solutions),
                "享受政策": random.choice(mock_policies),
                "使用时长": f"{random.randint(20, 80)}小时",
            }
    
    logger.info("[_generate_mock_data] 已生成模拟数据")


@tool(description="标记当前对话为报告生成场景，触发系统自动切换至报告专用提示词模板")
def fill_context_for_report() -> str:
    """
    无入参，调用后触发中间件自动为报告生成场景动态注入上下文信息
    
    使用场景：
    - 用户明确要求生成使用报告
    - 用户查询个人项目进展报告
    - 用户需要平台运营统计数据
    
    返回：
        确认字符串 "fill_context_for_report已调用"
    
    注意：
        此工具本身不返回报告内容，而是通过中间件机制切换提示词模板，
        后续的工具调用和模型生成将使用报告专用的提示词
    """
    return "fill_context_for_report已调用"
