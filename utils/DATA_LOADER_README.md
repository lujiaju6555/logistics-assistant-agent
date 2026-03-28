# 数据加载器使用说明

## 概述
新的数据加载器 (`data_loader.py`) 已经完全基于 `./data/database` 目录下的真实数据库文件（OMS.xlsx, TMS.xlsx, WMS.xlsx），不再包含任何mock业务数据。

## 主要功能

### 1. 数据加载
- **订单数据**: 从OMS.xlsx加载订单信息
- **运输信息**: 从TMS.xlsx加载运输详情
- **仓库信息**: 从WMS.xlsx加载仓储数据

### 2. 核心方法

#### 订单相关
- `get_all_orders()`: 获取所有订单
- `get_order_by_id(order_id)`: 根据订单ID获取详情
- `get_orders_by_user(user_id)`: 获取指定用户的所有订单
- `get_order_status_summary()`: 获取订单状态统计

#### 运输相关
- `get_all_transport()`: 获取所有运输信息
- `get_transport_by_id(transport_id)`: 根据运单ID获取详情
- `get_transport_by_order_id(order_id)`: 根据订单ID获取对应运输信息
- `get_transport_status_summary()`: 获取运输状态统计

#### 仓库相关
- `get_all_warehouse_info()`: 获取所有仓库信息
- `get_warehouse_info_by_order_id(order_id)`: 根据订单ID获取对应仓库信息

#### 对话历史
- `get_chat_history(user_id)`: 获取用户对话历史
- `save_chat_history(user_id, chat_history)`: 保存用户对话历史

### 3. 数据缓存
- 使用内存缓存提高查询性能
- 可通过 `refresh_cache()` 方法刷新缓存

## 使用示例

```python
from utils.data_loader import data_loader

# 获取所有订单
all_orders = data_loader.get_all_orders()

# 获取单个订单详情
order = data_loader.get_order_by_id("JDL20241120001")

# 获取用户的所有订单
user_orders = data_loader.get_orders_by_user("U00189625")

# 获取订单对应的运输信息
transport = data_loader.get_transport_by_order_id("JDL20241120001")

# 获取订单状态统计
status_stats = data_loader.get_order_status_summary()
```

## 数据结构

### 订单数据包含字段
- order_id: 订单ID
- user_id: 用户ID
- order_type: 订单类型
- sender_name/phone/address: 寄件人信息
- receiver_name/phone/address: 收件人信息
- product: 商品
- order_time: 下单时间
- order_status: 订单状态
- payment_method: 支付方式
- delivery_requirements: 配送要求
- expected_delivery_time: 预计时效
- actual_payment: 实际支付金额
- linked_oms_id: 关联OMS订单ID
- transport_id: 运单ID

### 运输数据包含字段
- transport_id: 运单ID
- linked_oms_id: 关联OMS订单ID
- transport_method: 运输方式
- origin_station: 起始站点
- destination_station: 目的站点
- transport_status: 运输状态
- vehicle_number: 运输车辆编号
- departure_time: 发车时间
- estimated_arrival_time: 预计到达时间
- current_location: 当前位置

### 仓库数据包含字段
- warehouse_id: 仓储ID
- warehouse_name: 仓库名称
- warehouse_location: 仓库位置
- product_sku: 商品SKU
- linked_oms_id: 关联OMS订单ID
- storage_quantity: 存储数量
- storage_location: 库位编号
- inbound_time: 入库时间
- outbound_time: 出库时间
- warehouse_status: 仓库状态