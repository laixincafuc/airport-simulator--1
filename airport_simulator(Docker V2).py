import json
import csv
from datetime import datetime, timedelta
import random
import os
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# ====================== 配置常量 ======================

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# ====================== 模型核心类 ======================

class AirportResource:
    """机场资源类，表示关键运行资源"""
    def __init__(self, runway_capacity=30, gate_capacity=25, security_capacity=300, baggage_capacity=200):
        # 初始资源容量
        self.runway_capacity = runway_capacity  # 跑道容量 (架次/小时)
        self.gate_capacity = gate_capacity      # 停机位容量 (同时服务航班数)
        self.security_capacity = security_capacity  # 安检通道容量 (人/小时)
        self.baggage_capacity = baggage_capacity   # 行李处理容量 (件/小时)
        
        # 记录原始容量，用于重置
        self.original_runway_capacity = self.runway_capacity
        self.original_gate_capacity = self.gate_capacity
        self.original_security_capacity = self.security_capacity
        self.original_baggage_capacity = self.baggage_capacity
    
    def reset(self):
        """重置资源到初始状态"""
        self.runway_capacity = self.original_runway_capacity
        self.gate_capacity = self.original_gate_capacity
        self.security_capacity = self.original_security_capacity
        self.baggage_capacity = self.original_baggage_capacity
    
    def update_capacity(self, runway_capacity=None, gate_capacity=None, 
                       security_capacity=None, baggage_capacity=None):
        """更新资源容量"""
        if runway_capacity is not None:
            self.runway_capacity = runway_capacity
            self.original_runway_capacity = runway_capacity
        if gate_capacity is not None:
            self.gate_capacity = gate_capacity
            self.original_gate_capacity = gate_capacity
        if security_capacity is not None:
            self.security_capacity = security_capacity
            self.original_security_capacity = security_capacity
        if baggage_capacity is not None:
            self.baggage_capacity = baggage_capacity
            self.original_baggage_capacity = baggage_capacity
    
    def __str__(self):
        """返回资源状态字符串"""
        return (f"跑道容量: {self.runway_capacity}架次/小时, "
                f"停机位: {self.gate_capacity}个, "
                f"安检: {self.security_capacity}人/小时, "
                f"行李处理: {self.baggage_capacity}件/小时")

class Flight:
    """航班类，表示单个航班信息"""
    def __init__(self, flight_id, scheduled_departure, passengers=150, cargo=5.0):
        self.id = flight_id
        self.scheduled_dep = scheduled_departure  # 计划起飞时间 (datetime)
        self.actual_dep = scheduled_departure     # 实际起飞时间 (datetime)
        self.delay = 0                            # 总延误时间 (分钟)
        self.status = "计划中"                    # 航班状态
        self.passengers = passengers              # 乘客数量
        self.actual_passengers = passengers       # 实际登机乘客数
        self.cargo = cargo                        # 货物吨数
        self.actual_cargo = cargo                 # 实际装载货物吨数
        self.delay_reasons = []                   # 延误原因记录
    
    def add_delay(self, minutes, reason):
        """添加延误并记录原因"""
        self.delay += minutes
        self.actual_dep += timedelta(minutes=minutes)
        self.delay_reasons.append((reason, minutes))
    
    def __str__(self):
        """返回航班信息字符串"""
        return (f"航班 {self.id}: 计划 {self.scheduled_dep.strftime('%H:%M')} "
                f"实际 {self.actual_dep.strftime('%H:%M')} "
                f"延误 {self.delay}分钟")

class AirportEfficiencyModel:
    """机场效能评估主模型"""
    
    def __init__(self, airport_config=None):
        # 使用配置初始化资源，如果没有配置则使用默认值
        if airport_config:
            self.resources = AirportResource(
                runway_capacity=airport_config.get('runway_capacity', 30),
                gate_capacity=airport_config.get('gate_capacity', 25),
                security_capacity=airport_config.get('security_capacity', 300),
                baggage_capacity=airport_config.get('baggage_capacity', 200)
            )
        else:
            self.resources = AirportResource()
            
        self.flights = []
        self.events = []
        self.results = {}
    
    def update_airport_config(self, airport_config):
        """更新机场配置"""
        self.resources.update_capacity(
            runway_capacity=airport_config.get('runway_capacity'),
            gate_capacity=airport_config.get('gate_capacity'),
            security_capacity=airport_config.get('security_capacity'),
            baggage_capacity=airport_config.get('baggage_capacity')
        )

    def load_flights_from_csv(self, file_path):
        """从CSV文件加载航班计划（兼容 utf-8-sig/BOM、基础校验）"""
        try:
            flights = []
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                # 规整表头
                if reader.fieldnames:
                    reader.fieldnames = [(h or "").strip().lstrip('\ufeff') for h in reader.fieldnames]
                required = {'flight_id', 'scheduled_departure', 'passengers', 'cargo'}
                if not required.issubset(set(reader.fieldnames or [])):
                    missing = required - set(reader.fieldnames or [])
                    return False, f"加载航班失败：缺少列 {missing}，请检查表头是否为 {sorted(required)}"
                for row in reader:
                    # 基础清洗
                    fid = (row.get('flight_id') or '').strip()
                    tstr = (row.get('scheduled_departure') or '').strip()
                    pax = row.get('passengers')
                    cg = row.get('cargo')
                    if not fid or not tstr:
                        continue
                    # 解析时间
                    dep_time = datetime.strptime(tstr, '%H:%M').replace(year=2023, month=1, day=1)
                    # 强制类型
                    passengers = int(float(pax)) if pax not in (None, '') else 150
                    cargo = float(cg) if cg not in (None, '') else 5.0
                    flights.append(Flight(fid, dep_time, passengers, cargo))

            flights.sort(key=lambda x: x.scheduled_dep)
            self.flights = flights
            return True, f"成功加载 {len(self.flights)} 个航班"
        except Exception as e:
            return False, f"加载航班失败: {str(e)}"

    def generate_flights_schedule(self, daily_flights, start_hour=8, end_hour=22):
        """生成每日航班计划"""
        flights = []
        base_date = datetime(2023, 1, 1, start_hour, 0)
        
        # 计算可用时间窗口（分钟）
        total_minutes = (end_hour - start_hour) * 60
        
        # 平均分配航班时间
        interval = total_minutes / daily_flights
        
        for i in range(daily_flights):
            flight_id = f"CA{1000 + i}"
            
            # 计算起飞时间，加入随机波动
            scheduled_minutes = int(i * interval + random.uniform(-10, 10))
            scheduled_time = base_date + timedelta(minutes=max(0, scheduled_minutes))
            
            # 随机生成乘客和货物数据
            passengers = random.randint(100, 200)
            cargo = round(random.uniform(3.0, 8.0), 1)
            
            flight = Flight(flight_id, scheduled_time, passengers, cargo)
            flights.append(flight)
        
        # 按计划起飞时间排序
        flights.sort(key=lambda x: x.scheduled_dep)
        self.flights = flights
        return True, f"成功生成 {len(self.flights)} 个航班"
    
    def apply_event_to_resources(self, event):
        """应用事件影响至机场资源"""
        event_type = event["type"]
        
        if event_type == "runway_closed":
            # 跑道关闭事件
            runways_total = event.get("runways_total", 2)  # 从事件参数获取总跑道数
            runways_available = event["params"]["runways_available"]
            self.resources.runway_capacity = self.resources.original_runway_capacity * (runways_available / runways_total)
            return f"跑道关闭 → 容量降至 {self.resources.runway_capacity:.1f} 架次/小时"
        
        elif event_type == "airline_delay":
            # 航空公司延误事件 - 不影响资源容量，将在航班处理时应用
            return f"航空公司延误 → {event['params']['delay_rate']*100}%航班平均延误{event['params']['avg_delay']}分钟"
        
        elif event_type == "airport_resource_failure":
            # 机场资源失效事件（整合安检、行李、停机位故障）
            reduction = event["params"]["capacity_reduction"]  # 使用固定值0.9
            self.resources.security_capacity = self.resources.original_security_capacity * (1 - reduction)
            self.resources.baggage_capacity = self.resources.original_baggage_capacity * (1 - reduction)
            self.resources.gate_capacity = max(1, int(self.resources.original_gate_capacity * (1 - reduction)))
            return f"机场资源失效 → 安检容量降至 {self.resources.security_capacity:.1f} 人/小时, 行李处理降至 {self.resources.baggage_capacity:.1f} 件/小时, 停机位降至 {self.resources.gate_capacity} 个"
        
        return f"事件类型 '{event_type}' 已记录"
    
    def calculate_queue_delay(self, arrival_rate, service_capacity):
        """计算排队系统延误时间"""
        if service_capacity <= 0:
            return 60  # 系统完全崩溃
        
        utilization = arrival_rate / service_capacity
        if utilization >= 1:
            # 系统过载，使用最大延误
            return 60
        elif utilization < 0.7:
            # 低负载时使用简化公式
            return (utilization ** 2) * 10
        else:
            # 高负载时使用Kingman公式
            avg_service_time = 60 / service_capacity  # 分钟/单位
            return (utilization / (1 - utilization)) * avg_service_time
    
    def apply_initial_delays(self, flights, events, current_time):
        """应用初始延误（如航空公司原因造成的延误）"""
        for event in events:
            if event["type"] != "airline_delay":
                continue
            
            # 解析事件时间
            start_time = datetime.strptime(event["start"], '%H:%M').replace(year=2023, month=1, day=1)
            end_time = datetime.strptime(event["end"], '%H:%M').replace(year=2023, month=1, day=1)
            
            # 检查当前时间是否在事件影响期内
            if start_time <= current_time < end_time:
                delay_rate = event["params"]["delay_rate"]
                avg_delay = event["params"]["avg_delay"]
                
                # 随机选择部分航班添加初始延误
                for flight in flights:
                    if flight.scheduled_dep >= start_time and flight.scheduled_dep < end_time:
                        if random.random() < delay_rate:
                            flight.add_delay(avg_delay, "航空公司初始延误")
        
        return flights
    
    def simulate_flight_processing(self, flights, resources, start_time, end_time):
        """模拟航班处理流程"""
        current_time = start_time
        
        # 按时间顺序处理航班
        for flight in flights:
            # 跳过不在当前时间段的航班
            if flight.scheduled_dep < start_time or flight.scheduled_dep >= end_time:
                continue
            
            # 更新当前时间
            current_time = flight.scheduled_dep
            
            # 1. 值机与安检流程（乘客级）
            # 假设乘客在航班起飞前60-120分钟到达
            pax_arrival_window = random.randint(60, 120)
            pax_arrival_rate = flight.passengers / (pax_arrival_window / 60)  # 乘客/小时
            
            # 计算安检延误
            security_delay = self.calculate_queue_delay(pax_arrival_rate, resources.security_capacity)
            if security_delay > 0:
                flight.add_delay(security_delay, "安检排队")
            
            # 2. 行李处理流程
            # 假设每名乘客有1.2件行李
            baggage_count = flight.passengers * 1.2
            baggage_arrival_rate = baggage_count / (pax_arrival_window / 60)  # 行李/小时
            baggage_delay = self.calculate_queue_delay(baggage_arrival_rate, resources.baggage_capacity)
            if baggage_delay > 0:
                flight.add_delay(baggage_delay, "行李处理")
            
            # 3. 跑道起飞排队（航班级）
            # 计算当前小时内的航班数量（简单估算到达率）
            hourly_flights = [f for f in flights 
                            if f.scheduled_dep.hour == current_time.hour]
            flight_arrival_rate = len(hourly_flights)  # 航班/小时
            
            # 计算跑道延误
            runway_delay = self.calculate_queue_delay(flight_arrival_rate, resources.runway_capacity)
            if runway_delay > 0:
                flight.add_delay(runway_delay, "跑道排队")
            
            # 更新航班状态
            flight.status = "已起飞"
        
        return flights
    
    def calculate_passenger_loss(self, flights):
        """计算客运吞吐量损失"""
        total_planned = sum(f.passengers for f in flights)
        total_actual = 0
        
        for flight in flights:
            # 延误导致乘客流失：每10分钟延误流失3%的乘客
            loss_ratio = min(0.3, flight.delay / 10 * 0.03)
            flight.actual_passengers = flight.passengers * (1 - loss_ratio)
            total_actual += flight.actual_passengers
        
        return total_actual, total_planned, total_planned - total_actual
    
    def calculate_cargo_loss(self, flights):
        """计算货运吞吐量损失"""
        total_planned = sum(f.cargo for f in flights)
        total_actual = 0
        
        for flight in flights:
            # 货运损失：每15分钟延误流失2%的货物
            loss_ratio = min(0.2, flight.delay / 15 * 0.02)
            flight.actual_cargo = flight.cargo * (1 - loss_ratio)
            total_actual += flight.actual_cargo
        
        return total_actual, total_planned, total_planned - total_actual
    
    def calculate_airline_loss(self, flights):
        """计算航空公司经济损失"""
        total_loss = 0
        
        for flight in flights:
            # 延误成本：每分钟1500元
            delay_cost = flight.delay * 1500 / 10000  # 转换为万元
            
            # 乘客流失成本：每位流失乘客损失2000元
            passenger_loss = (flight.passengers - flight.actual_passengers) * 2000 / 10000
            
            # 货物损失成本：每吨货物损失50000元
            cargo_loss = (flight.cargo - flight.actual_cargo) * 50000 / 10000
            
            total_loss += delay_cost + passenger_loss + cargo_loss
        
        return total_loss
    
    def calculate_airport_loss(self, flights, resources, duration_hours):
        """计算机场经济损失"""
        # 1. 起降费损失（延误导致起降次数减少）
        planned_flights = len(flights)
        actual_flights = len([f for f in flights if f.status == "已起飞"])
        landing_fee_loss = max(0, (planned_flights - actual_flights)) * 2.0  # 假设每架次起降费2万元
        
        # 2. 资源闲置成本
        # 跑道闲置成本（假设正常利用率80%）
        if resources.runway_capacity * duration_hours > 0:
            runway_utilization = actual_flights / (resources.runway_capacity * duration_hours)
            runway_idle_cost = max(0, (0.8 - min(0.8, runway_utilization))) * 10 * duration_hours
        else:
            runway_utilization = 0
            runway_idle_cost = 0
        
        # 3. 额外运营成本（人工、设备等）
        operational_cost = sum(f.delay for f in flights) * 100 / 60 / 10000  # 每分钟100元
        
        total_loss = landing_fee_loss + runway_idle_cost + operational_cost
        return total_loss
    
    def simulate(self, start_time_str, end_time_str):
        """运行模拟"""
        # 重置资源
        self.resources.reset()
        
        # 转换时间格式
        start_time = datetime.strptime(start_time_str, '%H:%M').replace(year=2023, month=1, day=1)
        end_time = datetime.strptime(end_time_str, '%H:%M').replace(year=2023, month=1, day=1)
        
        # 应用事件影响
        event_logs = []
        for event in self.events:
            event_start = datetime.strptime(event["start"], '%H:%M').replace(year=2023, month=1, day=1)
            event_end = datetime.strptime(event["end"], '%H:%M').replace(year=2023, month=1, day=1)
            
            # 只应用在模拟时间段内的事件
            if event_end > start_time and event_start < end_time:
                log = self.apply_event_to_resources(event)
                event_logs.append(log)
        
        # 应用初始延误
        self.apply_initial_delays(self.flights, self.events, start_time)
        
        # 模拟航班处理
        self.simulate_flight_processing(self.flights, self.resources, start_time, end_time)
        
        # 计算效能指标
        passenger_results = self.calculate_passenger_loss(self.flights)
        cargo_results = self.calculate_cargo_loss(self.flights)
        airline_loss = self.calculate_airline_loss(self.flights)
        duration_hours = (end_time - start_time).total_seconds() / 3600
        airport_loss = self.calculate_airport_loss(self.flights, self.resources, duration_hours)
        
        # 汇总结果
        self.results = {
            'passenger': passenger_results,
            'cargo': cargo_results,
            'airline_loss': airline_loss,
            'airport_loss': airport_loss,
            'event_logs': event_logs,
            'flights': self.flights,
            'duration_hours': duration_hours
        }
        
        return self.results

# ====================== Flask API 服务 ======================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "service": "Airport Efficiency Model API",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/upload/flights', methods=['POST'])
def upload_flights_file():
    """上传航班CSV文件接口"""
    try:
        # 检查是否有文件被上传
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "没有上传文件"
            }), 400
        
        file = request.files['file']
        
        # 检查文件名是否为空
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "未选择文件"
            }), 400
        
        # 检查文件类型
        if file and allowed_file(file.filename):
            # 安全保存文件名
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 验证CSV文件格式
            try:
                model = AirportEfficiencyModel()
                success, message = model.load_flights_from_csv(file_path)
                
                if success:
                    return jsonify({
                        "success": True,
                        "message": message,
                        "filename": filename,
                        "file_path": file_path,
                        "flights_count": len(model.flights)
                    })
                else:
                    # 删除无效文件
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return jsonify({
                        "success": False,
                        "error": f"文件格式验证失败: {message}"
                    }), 400
                    
            except Exception as e:
                # 删除无效文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({
                    "success": False,
                    "error": f"文件处理失败: {str(e)}"
                }), 500
        
        else:
            return jsonify({
                "success": False,
                "error": "不支持的文件类型，请上传CSV文件"
            }), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"上传过程发生错误: {str(e)}"
        }), 500

@app.route('/simulate', methods=['POST'])
def simulate_airport():
    """机场效能模拟接口"""
    try:
        # 获取输入参数
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "请求体必须为JSON格式"
            }), 400
        
        # 验证必需参数
        required_params = ['daily_flights', 'simulation_start', 'simulation_end']
        missing_params = [param for param in required_params if param not in data]
        if missing_params:
            return jsonify({
                "success": False,
                "error": f"缺少必需参数: {missing_params}"
            }), 400
        
        # 创建模型实例
        airport_config = {
            'daily_flights': data.get('daily_flights', 300),
            'runway_capacity': data.get('runway_capacity', 30),
            'gate_capacity': data.get('gate_capacity', 25),
            'security_capacity': data.get('security_capacity', 300),
            'baggage_capacity': data.get('baggage_capacity', 200),
            'runways_total': data.get('runways_total', 2)
        }
        
        model = AirportEfficiencyModel(airport_config)
        
        # 设置模拟时间
        simulation_start = f"{data['simulation_start']}:00"
        simulation_end = f"{data['simulation_end']}:00"
        
        # 处理航班数据
        if 'flights_file' in data:
            # 从文件加载航班数据
            file_path = data['flights_file']
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(file_path):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(file_path))
            
            success, message = model.load_flights_from_csv(file_path)
            if not success:
                return jsonify({
                    "success": False,
                    "error": f"航班数据加载失败: {message}"
                }), 400
        else:
            # 生成航班计划
            success, message = model.generate_flights_schedule(
                data['daily_flights'],
                data['simulation_start'],
                data['simulation_end']
            )
            if not success:
                return jsonify({
                    "success": False,
                    "error": f"航班计划生成失败: {message}"
                }), 400
        
        # 处理事件数据
        if 'events' in data:
            for event_data in data['events']:
                # 验证事件必需参数
                event_required = ['type', 'start', 'end']
                event_missing = [param for param in event_required if param not in event_data]
                if event_missing:
                    return jsonify({
                        "success": False,
                        "error": f"事件缺少必需参数: {event_missing}"
                    }), 400
                
                # 构建事件对象
                event = {
                    "type": event_data['type'],
                    "start": event_data['start'],
                    "end": event_data['end'],
                    "params": event_data.get('params', {})
                }
                
                # 对于跑道关闭事件，添加总跑道数信息
                if event_data['type'] == "runway_closed":
                    event["runways_total"] = airport_config['runways_total']
                
                model.events.append(event)
        
        # 运行模拟
        results = model.simulate(simulation_start, simulation_end)
        
        # 构建输出响应
        output = {
            "success": True,
            "results": {
                "影响时长_小时": results['duration_hours'],
                "影响旅客吞吐量_人次": results['passenger'][2],  # 损失量
                "影响货邮吞吐量_公吨": results['cargo'][2],     # 损失量
                "机场经济损失_万元": results['airport_loss'],
                "航空公司经济损失_万元": results['airline_loss']
            },
            "details": {
                "客运吞吐量": {
                    "计划": results['passenger'][1],
                    "实际": results['passenger'][0],
                    "损失": results['passenger'][2]
                },
                "货运吞吐量": {
                    "计划": results['cargo'][1],
                    "实际": results['cargo'][0],
                    "损失": results['cargo'][2]
                },
                "事件影响": results['event_logs'],
                "航班统计": {
                    "总航班数": len(results['flights']),
                    "延误航班数": len([f for f in results['flights'] if f.delay > 0]),
                    "总延误时间_分钟": sum(f.delay for f in results['flights'])
                }
            }
        }
        
        return jsonify(output)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"模拟过程发生错误: {str(e)}"
        }), 500

@app.route('/config', methods=['GET'])
def get_default_config():
    """获取默认配置接口"""
    default_config = {
        "daily_flights": 300,
        "runway_capacity": 30,
        "gate_capacity": 25,
        "security_capacity": 300,
        "baggage_capacity": 200,
        "runways_total": 2,
        "simulation_start": 8,
        "simulation_end": 22
    }
    
    event_types = {
        "runway_closed": {
            "description": "跑道关闭事件",
            "required_params": ["runways_available"],
            "params_description": {
                "runways_available": "可用跑道数量（整数）"
            }
        },
        "airline_delay": {
            "description": "航空公司延误事件",
            "required_params": ["delay_rate", "avg_delay"],
            "params_description": {
                "delay_rate": "延误比例（0-1之间的小数）",
                "avg_delay": "平均延误时间（分钟，整数）"
            }
        },
        "airport_resource_failure": {
            "description": "机场资源失效事件",
            "required_params": ["capacity_reduction"],
            "params_description": {
                "capacity_reduction": "容量减少比例（默认0.9）"
            }
        }
    }
    
    return jsonify({
        "default_config": default_config,
        "event_types": event_types
    })

@app.route('/files/uploaded', methods=['GET'])
def list_uploaded_files():
    """获取已上传文件列表"""
    try:
        files = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path) and allowed_file(filename):
                file_info = {
                    'filename': filename,
                    'file_path': file_path,
                    'size': os.path.getsize(file_path),
                    'upload_time': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                }
                files.append(file_info)
        
        return jsonify({
            "success": True,
            "files": files
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"获取文件列表失败: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)