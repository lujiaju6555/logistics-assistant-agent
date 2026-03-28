import json
import os
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime


class DataLoader:
    """数据加载器 - 从实际数据库文件加载数据"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.database_dir = os.path.join(data_dir, "database")
        self.data_cache = {}
        self.load_orders()
        self.load_transport_info()
        self.load_warehouse_info()
        
    def load_orders(self) -> Dict[str, Any]:
        """从OMS.xlsx加载订单数据"""
        if "orders" in self.data_cache:
            return self.data_cache["orders"]
            
        orders = {}
        oms_path = os.path.join(self.database_dir, "OMS.xlsx")
        
        if os.path.exists(oms_path):
            try:
                df = pd.read_excel(oms_path)
                for _, row in df.iterrows():
                    order_id = str(row.get("订单 ID", ""))
                    if order_id:
                        orders[order_id] = {
                            "order_id": order_id,
                            "user_id": str(row.get("用户 ID", "")),
                            "order_type": str(row.get("订单类型", "")),
                            "sender_name": str(row.get("寄件人姓名", "")),
                            "sender_phone": str(row.get("寄件人电话", "")),
                            "sender_address": str(row.get("寄件人地址", "")),
                            "receiver_name": str(row.get("收件人姓名", "")),
                            "receiver_phone": str(row.get("收件人电话", "")),
                            "receiver_address": str(row.get("收件人地址", "")),
                            "product": str(row.get("商品", "")),
                            "order_time": str(row.get("下单时间", "")),
                            "order_status": str(row.get("订单状态", "")),
                            "payment_method": str(row.get("支付方式", "")),
                            "delivery_requirements": str(row.get("配送要求说明", "")),
                            "expected_delivery_time": str(row.get("预计时效", "")),
                            "actual_payment": float(row.get("实际支付金额（元）", 0)),
                            "linked_oms_id": str(row.get("关联 OMS 订单 ID", "")),
                            "transport_id": str(row.get("运单 ID", ""))
                        }
            except Exception as e:
                print(f"读取OMS.xlsx失败: {e}")
                
        self.data_cache["orders"] = orders
        return orders

    def load_transport_info(self) -> Dict[str, Any]:
        """从TMS.xlsx加载运输信息"""
        if "transport" in self.data_cache:
            return self.data_cache["transport"]
            
        transport = {}
        tms_path = os.path.join(self.database_dir, "TMS.xlsx")
        
        if os.path.exists(tms_path):
            try:
                df = pd.read_excel(tms_path)
                for _, row in df.iterrows():
                    transport_id = str(row.get("运单 ID", ""))
                    if transport_id:
                        transport[transport_id] = {
                            "transport_id": transport_id,
                            "linked_oms_id": str(row.get("关联 OMS 订单 ID", "")),
                            "transport_method": str(row.get("运输方式", "")),
                            "origin_station": str(row.get("起始站点", "")),
                            "destination_station": str(row.get("目的站点", "")),
                            "transport_status": str(row.get("运输状态", "")),
                            "vehicle_number": str(row.get("运输车辆编号", "")),
                            "departure_time": str(row.get("发车时间", "")),
                            "estimated_arrival_time": str(row.get("预计到达时间", "")),
                            "current_location": str(row.get("当前位置", ""))
                        }
            except Exception as e:
                print(f"读取TMS.xlsx失败: {e}")
                
        self.data_cache["transport"] = transport
        return transport

    def load_warehouse_info(self) -> Dict[str, Any]:
        """从WMS.xlsx加载仓库信息"""
        if "warehouse" in self.data_cache:
            return self.data_cache["warehouse"]
            
        warehouse = {}
        wms_path = os.path.join(self.database_dir, "WMS.xlsx")
        
        if os.path.exists(wms_path):
            try:
                df = pd.read_excel(wms_path)
                for _, row in df.iterrows():
                    warehouse_id = str(row.get("仓储 ID", ""))
                    if warehouse_id:
                        warehouse[warehouse_id] = {
                            "warehouse_id": warehouse_id,
                            "warehouse_name": str(row.get("仓库名称", "")),
                            "warehouse_location": str(row.get("仓库位置", "")),
                            "product_sku": str(row.get("商品 SKU", "")),
                            "linked_oms_id": str(row.get("关联 OMS 订单 ID", "")),
                            "storage_quantity": int(row.get("存储数量", 0)),
                            "storage_location": str(row.get("库位编号", "")),
                            "inbound_time": str(row.get("入库时间", "")),
                            "outbound_time": str(row.get("出库时间", "")),
                            "warehouse_status": str(row.get("仓库状态", ""))
                        }
            except Exception as e:
                print(f"读取WMS.xlsx失败: {e}")
                
        self.data_cache["warehouse"] = warehouse
        return warehouse

    def get_order_by_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        """根据订单ID获取订单详情"""
        orders = self.load_orders()
        return orders.get(order_id)

    def get_transport_by_id(self, transport_id: str) -> Optional[Dict[str, Any]]:
        """根据运单ID获取运输信息"""
        transport = self.load_transport_info()
        return transport.get(transport_id)

    def get_orders_by_user(self, user_id: str) -> list:
        """根据用户ID获取用户的所有订单"""
        orders = self.load_orders()
        return [order for order in orders.values() if order.get("user_id") == user_id]

    def get_transport_by_order_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        """根据订单ID获取对应的运输信息"""
        order = self.get_order_by_id(order_id)
        if order:
            transport_id = order.get("transport_id")
            if transport_id:
                return self.get_transport_by_id(transport_id)
        return None

    def get_warehouse_info_by_order_id(self, order_id: str) -> list:
        """根据订单ID获取对应的仓库信息"""
        warehouse = self.load_warehouse_info()
        return [wh for wh in warehouse.values() if wh.get("linked_oms_id") == order_id]

    def get_all_orders(self) -> Dict[str, Any]:
        """获取所有订单"""
        return self.load_orders()

    def get_all_transport(self) -> Dict[str, Any]:
        """获取所有运输信息"""
        return self.load_transport_info()

    def get_all_warehouse_info(self) -> Dict[str, Any]:
        """获取所有仓库信息"""
        return self.load_warehouse_info()

    def refresh_cache(self):
        """刷新数据缓存"""
        self.data_cache.clear()

    def save_chat_history(self, user_id: str, chat_history: list):
        """保存用户对话历史到独立文件"""
        chat_dir = os.path.join(self.data_dir, "chat_history")
        os.makedirs(chat_dir, exist_ok=True)
        
        chat_file = os.path.join(chat_dir, f"{user_id}.json")
        with open(chat_file, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)

    def get_chat_history(self, user_id: str) -> list:
        """获取用户对话历史"""
        chat_file = os.path.join(self.data_dir, "chat_history", f"{user_id}.json")
        if os.path.exists(chat_file):
            try:
                with open(chat_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"读取对话历史失败: {e}")
        return []
    
    def save_user_structured_info(self, user_id: str, structured_info: dict):
        """保存用户结构化信息"""
        info_dir = os.path.join(self.data_dir, "user_structured_info")
        os.makedirs(info_dir, exist_ok=True)
        
        info_file = os.path.join(info_dir, f"{user_id}.json")
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(structured_info, f, ensure_ascii=False, indent=2)
    
    def get_user_structured_info(self, user_id: str) -> dict:
        """获取用户结构化信息"""
        info_file = os.path.join(self.data_dir, "user_structured_info", f"{user_id}.json")
        if os.path.exists(info_file):
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"读取用户结构化信息失败: {e}")
        return {}

    def get_order_status_summary(self) -> Dict[str, int]:
        """获取订单状态统计"""
        orders = self.load_orders()
        status_count = {}
        for order in orders.values():
            status = order.get("order_status", "未知")
            status_count[status] = status_count.get(status, 0) + 1
        return status_count

    def get_transport_status_summary(self) -> Dict[str, int]:
        """获取运输状态统计"""
        transport = self.load_transport_info()
        status_count = {}
        for trans in transport.values():
            status = trans.get("transport_status", "未知")
            status_count[status] = status_count.get(status, 0) + 1
        return status_count


# 创建全局数据加载器实例
data_loader = DataLoader()