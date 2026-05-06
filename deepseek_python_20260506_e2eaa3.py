#!/usr/bin/env python3
"""
【MVP】智能家居故障诊断与自愈Agent
基于LangGraph + MiMo API（兼容OpenAI格式）

使用前请确保已安装依赖：pip install langgraph openai

模拟模式说明（推荐先选）：
- 设置 MOCK_MODE=true 可使用内置模拟数据演示完整流程（无需API Key）
- 或配置 MIMO_API_KEY 连接真实MiMo API进行推理测试
"""

import json
import logging
import os
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional, Literal

# ==================== 依赖检查与配置 ====================

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    raise ImportError("请先运行: pip install langgraph")

# 可选依赖：连接真实MiMo API时需要
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: openai 库未安装，将强制使用模拟模式。安装命令: pip install openai")


# ==================== 配置 ====================

# 模拟模式（推荐先使用模拟模式体验完整流程）
MOCK_MODE = os.environ.get("MOCK_MODE", "true").lower() == "true"
# MiMo API 配置（如需真实测试，请配置以下变量）
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2-pro"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==================== MiMo API 调用封装 ====================

def call_mimo(prompt: str, system_prompt: str = None) -> str:
    """
    调用MiMo API，支持模拟模式
    返回模型的回复内容
    """
    if MOCK_MODE or not OPENAI_AVAILABLE or not MIMO_API_KEY:
        logger.info("使用模拟模式进行诊断推理...")
        return _mock_diagnosis(prompt)
    
    try:
        client = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=MIMO_MODEL,
            messages=messages,
            temperature=0.1,  # 诊断场景要求确定性输出
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"MiMo API调用失败: {e}，回退模拟模式")
        return _mock_diagnosis(prompt)


def _mock_diagnosis(prompt: str) -> str:
    """模拟诊断推理结果"""
    return """
【推理链】
1. 异常识别：设备最后在线时间为前日，当前状态unknown，传感器读数显示电池电量仅剩12%。
2. 范围缩小：同网关下其他设备工作正常，排除网关故障；该门窗传感器为蓝牙Mesh通讯，依赖主网关连接。
3. 依赖链排查：网关在线、固件版本为当前最新，排除固件兼容性问题。
4. 逐层聚焦：设备自身电池电量从30%下降至12%，未见明显异常波动；低电量会导致蓝牙信号不稳定，造成间歇性离线后最终彻底失联。
5. 根因定位：判断为电池耗尽，设备因供电不足而离线。

【根因结论】
门窗传感器电池电量不足，导致设备离线。

【故障类型】
供电/电量

【置信度】
92%
"""


# ==================== 模拟设备数据 ====================

MOCK_DEVICES = {
    "mock_sensor_001": {
        "did": "mock_sensor_001",
        "name": "客厅门窗传感器",
        "model": "mijia.sensor.door",
        "status": "offline",
        "props": {"battery": 12, "last_seen": "2026-05-04T08:30:00"},
        "signal": -85
    },
    "mock_sensor_002": {
        "did": "mock_sensor_002",
        "name": "卧室人体传感器",
        "model": "xiaomi.sensor_occupancy",
        "status": "online",
        "props": {"occupancy": "unknown", "last_seen": "2026-05-06T10:00:00"},
        "signal": -45
    },
    "mock_plug_001": {
        "did": "mock_plug_001",
        "name": "客厅智能插座",
        "model": "mi.smart_plug",
        "status": "online",
        "props": {"power": 0},
        "signal": -30
    },
    "mock_gateway_001": {
        "did": "mock_gateway_001",
        "name": "中枢网关",
        "model": "xiaomi.gateway.central",
        "status": "online",
        "props": {"firmware": "1.5.0_0026", "child_device_count": 3},
        "signal": -20
    }
}


def get_mock_device_status(device_id: str) -> Dict:
    """获取模拟设备状态"""
    if device_id in MOCK_DEVICES:
        return MOCK_DEVICES[device_id]
    return {"did": device_id, "status": "unknown", "error": "device not found"}


def get_mock_devices_list() -> List[Dict]:
    """获取所有模拟设备列表"""
    return list(MOCK_DEVICES.values())


# ==================== Agent 状态定义 ====================

class AgentState(TypedDict):
    """Agent工作流状态"""
    # 输入
    query: str
    failed_device_id: str
    failure_time: str
    # 中间状态
    device_info: Dict[str, Any]
    diagnostic_chain: str
    root_cause: Dict[str, Any]
    healing_actions: List[str]
    # 输出
    report: str
    status: Literal["success", "partial", "failed"]
    requires_human: bool


# ==================== Agent 节点定义 ====================

def get_device_info(state: AgentState) -> AgentState:
    """获取设备详细信息"""
    device_id = state["failed_device_id"]
    # 获取设备状态（可替换为真实的米家API调用）
    device = get_mock_device_status(device_id)
    state["device_info"] = device
    logger.info(f"设备信息获取成功: {device.get('name')} ({device_id})")
    return state


def diagnose(state: AgentState) -> AgentState:
    """诊断Agent - 长链推理"""
    device = state["device_info"]
    
    # 构建诊断上下文
    context = f"""
设备ID: {device.get('did')}
设备名称: {device.get('name')}
设备型号: {device.get('model')}
当前状态: {device.get('status')}
设备属性: {json.dumps(device.get('props', {}), ensure_ascii=False)}
信号强度: {device.get('signal', 'unknown')} dBm
"""
    # 构建诊断提示词
    prompt = f"""
请基于以下设备信息进行故障诊断。

设备信息：
{context}

请按照以下步骤进行长链推理：
1. 异常识别：描述观察到的异常现象
2. 范围缩小：判断是单一设备问题还是共性问题
3. 依赖链排查：检查网关、网络、固件等依赖项
4. 逐层聚焦：逐层定位故障根因
5. 根因定位：给出最可能的根因及判断依据

输出格式要求：
- 推理链：[按步骤展示思考过程]
- 根因结论：[一句话总结根因]
- 故障类型：[硬件/软件/网络/供电/固件/配置]
- 置信度：[0-100%]
"""
    
    result = call_mimo(prompt, system_prompt="你是一个专业的物联网智能家居故障诊断专家")
    state["diagnostic_chain"] = result
    state["root_cause"] = _parse_root_cause(result)
    logger.info(f"诊断完成，根因: {state['root_cause'].get('description')}")
    return state


def _parse_root_cause(diagnosis: str) -> Dict:
    """解析诊断结果中的根因信息"""
    lines = diagnosis.split("\n")
    root_cause = {
        "description": "诊断分析中，详见报告",
        "type": "unknown",
        "confidence": 50
    }
    for line in lines:
        line_lower = line.lower()
        if "根因结论" in line or "根因：" in line:
            # 提取根因描述
            parts = line.split("：") if "：" in line else line.split(":")
            if len(parts) > 1:
                root_cause["description"] = parts[1].strip()
        elif "故障类型" in line_lower:
            # 提取故障类型
            if "供电" in line or "电量" in line:
                root_cause["type"] = "power"
            elif "网络" in line or "信号" in line:
                root_cause["type"] = "network"
            elif "硬件" in line:
                root_cause["type"] = "hardware"
            elif "固件" in line or "软件" in line:
                root_cause["type"] = "firmware"
        elif "置信度" in line_lower:
            # 提取置信度
            import re
            match = re.search(r"(\d+)%", line)
            if match:
                root_cause["confidence"] = int(match.group(1))
    return root_cause


def heal(state: AgentState) -> AgentState:
    """自愈Agent - 执行修复方案"""
    device = state["device_info"]
    root_cause = state["root_cause"]
    confidence = root_cause.get("confidence", 50)
    fault_type = root_cause.get("type", "unknown")
    
    actions = []
    requires_human = False
    
    # 置信度过低时请求人工介入
    if confidence < 70:
        actions.append(f"[待人工确认] 诊断置信度较低({confidence}%)，建议人工介入进一步排查")
        requires_human = True
    else:
        # 根据故障类型执行不同的自愈策略
        if fault_type == "power":
            actions.append(f"[自动] 检测到电量不足，已发送低电量通知到米家App")
            actions.append("[建议] 请更换设备电池")
        elif fault_type == "network":
            actions.append("[自动] 尝试重启网关以恢复网络连接... [模拟] 网关重启指令已发送")
        elif fault_type == "firmware":
            actions.append("[自动] 尝试远程重启设备以修复异常... [模拟] 设备重启指令已发送")
        elif fault_type == "hardware":
            actions.append("[待人工] 检测到硬件可能存在故障，建议联系售后或更换设备")
            requires_human = True
        else:
            actions.append("[待人工] 故障类型不明，建议人工排查")
            requires_human = True
    
    state["healing_actions"] = actions
    state["requires_human"] = requires_human
    
    if requires_human:
        state["status"] = "partial"
    else:
        state["status"] = "success"
    
    logger.info(f"自愈计划生成完成，共{len(actions)}条动作")
    return state


def generate_report(state: AgentState) -> AgentState:
    """报告Agent - 生成最终诊断报告"""
    device = state["device_info"]
    root_cause = state["root_cause"]
    
    # 构建卡片式报告
    report_lines = [
        "╔" + "═" * 58 + "╗",
        "║" + "        【智能家居设备诊断报告】        ".center(58) + "║",
        "╠" + "═" * 58 + "╣",
        f"║ 设备名称:{device.get('name', '未知'):<45}║",
        f"║ 设备ID:  {device.get('did', '未知'):<45}║",
        f"║ 诊断时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<45}║",
        "╠" + "═" * 58 + "╣",
        "",
        "📋 【智能诊断摘要】",
        "─" * 50,
        f"📉 当前状态: {device.get('status', 'unknown')}",
        f"🔍 根因定位: {root_cause.get('description')}",
        f"🏷️ 故障类型: {_translate_fault_type(root_cause.get('type'))}",
        f"📊 诊断置信度: {root_cause.get('confidence', 50)}%",
        "",
        "🧠 【诊断推理链】",
        "─" * 50,
        state["diagnostic_chain"][:500] + ("..." if len(state["diagnostic_chain"]) > 500 else ""),
        "",
        "🔧 【自愈动作执行结果】",
        "─" * 50,
    ]
    
    for action in state["healing_actions"]:
        report_lines.append(f"  • {action}")
    
    status_emoji = "✅" if state["status"] == "success" else "⚠️" if state["status"] == "partial" else "❌"
    status_text = "已自动修复" if state["status"] == "success" else "部分成功/需人工干预" if state["status"] == "partial" else "修复失败"
    report_lines.append("")
    report_lines.append(f"{status_emoji} 系统状态: {status_text}")
    
    if state["requires_human"]:
        report_lines.append("")
        report_lines.append("👤 人工干预提示")
        report_lines.append("─" * 50)
        report_lines.append("建议人工执行以下操作：1）确认故障设备现场状态；2）按上述诊断结论进行手动处理")
    
    report_lines.append("")
    report_lines.append("╚" + "═" * 58 + "╝")
    
    state["report"] = "\n".join(report_lines)
    logger.info("报告生成完成")
    return state


def _translate_fault_type(fault_type: str) -> str:
    """将故障类型代码转换为中文显示"""
    mapping = {
        "power": "🔋 供电/电量不足",
        "network": "📡 网络/通讯故障",
        "hardware": "🔧 硬件故障",
        "firmware": "💾 固件/软件异常",
        "configuration": "⚙️ 配置错误",
        "unknown": "❓ 待进一步诊断"
    }
    return mapping.get(fault_type, mapping["unknown"])


# ==================== 工作流编排 ====================

def build_workflow() -> StateGraph:
    """构建LangGraph工作流"""
    workflow = StateGraph(AgentState)
    workflow.add_node("get_device_info", get_device_info)
    workflow.add_node("diagnose", diagnose)
    workflow.add_node("heal", heal)
    workflow.add_node("generate_report", generate_report)
    
    workflow.set_entry_point("get_device_info")
    workflow.add_edge("get_device_info", "diagnose")
    workflow.add_edge("diagnose", "heal")
    workflow.add_edge("heal", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# ==================== 主程序 ====================

class SmartHomeFaultAgent:
    """智能家居故障诊断与自愈Agent主类"""
    
    def __init__(self):
        self.workflow = build_workflow()
        logger.info("智能家居故障诊断与自愈Agent 初始化完成")
    
    def diagnose_and_heal(self, device_id: str, query: Optional[str] = None) -> Dict:
        """
        对指定设备进行诊断和修复
        
        Args:
            device_id: 设备ID
            query: 可选的用户查询描述
            
        Returns:
            包含诊断结果的字典
        """
        initial_state: AgentState = {
            "query": query or f"请诊断设备 {device_id} 的故障",
            "failed_device_id": device_id,
            "failure_time": datetime.now().isoformat(),
            "device_info": {},
            "diagnostic_chain": "",
            "root_cause": {},
            "healing_actions": [],
            "report": "",
            "status": "failed",
            "requires_human": False
        }
        
        logger.info(f"开始诊断设备: {device_id}")
        final_state = self.workflow.invoke(initial_state)
        logger.info(f"诊断完成，最终状态: {final_state['status']}")
        return final_state
    
    def run_demo(self):
        """运行演示用例"""
        print("\n" + "=" * 60)
        print("  智能家居故障诊断与自愈Agent - MVP演示模式")
        print("=" * 60)
        print("\n[演示场景] 客厅门窗传感器离线问题\n")
        
        # 显示当前模拟模式状态
        if MOCK_MODE or not MIMO_API_KEY or not OPENAI_AVAILABLE:
            print("⚡ 运行模式: 模拟模式（使用内置诊断逻辑，无需API Key）\n")
        else:
            print("🚀 运行模式: 真实MiMo API模式\n")
        
        # 执行诊断与自愈
        result = self.diagnose_and_heal(
            device_id="mock_sensor_001",
            query="我家客厅的门窗传感器离线了，请问是什么原因？"
        )
        
        print(result["report"])
        
        return result


def main():
    """主入口"""
    agent = SmartHomeFaultAgent()
    result = agent.run_demo()
    return result


if __name__ == "__main__":
    main()