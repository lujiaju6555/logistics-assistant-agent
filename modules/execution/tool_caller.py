from typing import Dict, Any
from config import settings


class ToolCaller:
    """工具调用器"""

    def __init__(self):
        # 工具映射
        self.tool_map = {
            "query_logistics_price": self.query_logistics_price,
            "track_order_status": self.track_order_status,
            "create_shipment": self.create_shipment,
            "calculate_delivery_time": self.calculate_delivery_time
        }

    def get_tools(self) -> list:
        """获取工具列表"""
        return list(self.tool_map.keys())

    def call_tool(self, tool_name: str, parameters: Dict[str, Any], data_loader = None) -> str:
        """调用指定工具"""
        if tool_name in self.tool_map:
            try:
                # 为工具函数添加data_loader参数
                if tool_name == "track_order_status":
                    return self.tool_map[tool_name](parameters.get("order_id"), data_loader)
                elif tool_name == "query_logistics_price":
                    return self.tool_map[tool_name](
                        parameters.get("origin_city"),
                        parameters.get("dest_city"),
                        parameters.get("weight"),
                        data_loader
                    )
                elif tool_name == "calculate_delivery_time":
                    return self.tool_map[tool_name](
                        parameters.get("origin_city"),
                        parameters.get("dest_city"),
                        data_loader
                    )
                else:
                    return self.tool_map[tool_name](**parameters)
            except Exception as e:
                return f"工具调用失败: {str(e)}"
        else:
            return f"未知工具: {tool_name}"

    def route_and_call(self, question: str, extracted_params: Dict[str, Any], intent: Dict[str, Any] = None, data_loader = None) -> Dict[str, Any]:
        print(question)
        print(extracted_params)

        """路由并调用工具"""
        try:
            # 根据意图类型和子意图选择工具
            if not intent:
                return {
                    "tool_name": None,
                    "parameters": {},
                    "result": "未提供意图信息，无法选择工具"
                }

            intent_type = intent.get('intent_type', '')
            sub_intent = intent.get('sub_intent', '')

            # 根据意图选择工具和参数
            if intent_type == "BUSINESS_QUERY":
                if sub_intent == "PRICE_QUERY":
                    # 查询物流价格
                    origin_city = extracted_params.get('origin_city')
                    dest_city = extracted_params.get('dest_city')
                    weight = extracted_params.get('weight')
                    
                    if origin_city and dest_city and weight:
                        tool_name = "query_logistics_price"
                        parameters = {
                            "origin_city": origin_city,
                            "dest_city": dest_city,
                            "weight": weight
                        }
                    else:
                        return {
                            "tool_name": None,
                            "parameters": {},
                            "result": "缺少查询物流价格所需的参数: origin_city, dest_city, weight"
                        }
                
                elif sub_intent == "TRACKING":
                    # 查询订单状态
                    order_id = extracted_params.get('order_id')
                    
                    if order_id:
                        tool_name = "track_order_status"
                        parameters = {"order_id": order_id}
                    else:
                        return {
                            "tool_name": None,
                            "parameters": {},
                            "result": "缺少查询订单状态所需的参数: order_id"
                        }
                
                elif sub_intent == "DELIVERY_TIME":
                    # 计算预计送达时间
                    origin_city = extracted_params.get('origin_city')
                    dest_city = extracted_params.get('dest_city')
                    
                    if origin_city and dest_city:
                        tool_name = "calculate_delivery_time"
                        parameters = {
                            "origin_city": origin_city,
                            "dest_city": dest_city
                        }
                    else:
                        return {
                            "tool_name": None,
                            "parameters": {},
                            "result": "缺少计算送达时间所需的参数: origin_city, dest_city"
                        }
                
                elif sub_intent == "SHIPMENT":
                    # 创建物流订单
                    origin_city = extracted_params.get('origin_city')
                    dest_city = extracted_params.get('dest_city')
                    weight = extracted_params.get('weight')
                    service_type = extracted_params.get('service_type', 'standard')
                    
                    if origin_city and dest_city and weight:
                        tool_name = "create_shipment"
                        parameters = {
                            "origin_city": origin_city,
                            "dest_city": dest_city,
                            "weight": weight,
                            "service_type": service_type
                        }
                    else:
                        return {
                            "tool_name": None,
                            "parameters": {},
                            "result": "缺少创建物流订单所需的参数: origin_city, dest_city, weight"
                        }
                
                else:
                    return {
                        "tool_name": None,
                        "parameters": {},
                        "result": f"未知的子意图: {sub_intent}"
                    }
            
            else:
                return {
                    "tool_name": None,
                    "parameters": {},
                    "result": f"不支持的意图类型: {intent_type}"
                }

            # 调用工具，传递data_loader参数
            tool_result = self.call_tool(tool_name, parameters, data_loader)

            return {
                "tool_name": tool_name,
                "parameters": parameters,
                "result": tool_result
            }
        except Exception as e:
            return {
                "tool_name": None,
                "parameters": {},
                "result": f"路由失败: {str(e)}"
            }

    def query_logistics_price(self, origin_city: str, dest_city: str, weight: str, data_loader = None) -> str:
        """查询物流价格"""
        # 暂时使用默认价格，因为Excel文件中可能没有物流价格数据
        # 实际项目中可以从TMS.xlsx或其他数据源获取
        return f"从{origin_city}到{dest_city}，重量{weight}的物流价格为：12元"

    def track_order_status(self, order_id: str, data_loader = None) -> str:
        """查询订单状态"""
        if data_loader:
            order = data_loader.get_order_by_id(order_id)
            if order:
                transport_id = order.get("transport_id", "")
                transport_info = data_loader.get_transport_by_id(transport_id) if transport_id else None
                
                status = order.get("order_status", "未知")
                tracking_number = transport_id if transport_id else "无"
                
                if transport_info:
                    current_location = transport_info.get("current_location", "未知")
                    estimated_arrival = transport_info.get("estimated_arrival_time", "未知")
                    return f"订单{order_id}的状态为：{status}，物流单号：{tracking_number}，当前位置：{current_location}，预计到达时间：{estimated_arrival}"
                else:
                    return f"订单{order_id}的状态为：{status}，物流单号：{tracking_number}"
        return f"未找到订单{order_id}的状态"

    def create_shipment(self, origin_city: str, dest_city: str, weight: str, service_type: str) -> str:
        """创建物流订单"""
        # 模拟创建物流订单
        import random
        tracking_number = f"JD{random.randint(1000000000, 9999999999)}"
        return f"物流订单创建成功！\n出发地：{origin_city}\n目的地：{dest_city}\n重量：{weight}\n服务类型：{service_type}\n物流单号：{tracking_number}"

    def calculate_delivery_time(self, origin_city: str, dest_city: str, data_loader = None) -> str:
        """计算预计送达时间"""
        # 暂时使用默认送达时间，因为Excel文件中可能没有配送时间数据
        # 实际项目中可以从TMS.xlsx或其他数据源获取
        return f"从{origin_city}到{dest_city}的预计送达时间为：2天"


# 创建全局工具调用器实例
tool_caller = ToolCaller()