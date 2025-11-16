# get_weather.py
import random
from datetime import datetime, timedelta
import json

from tool_server.utils.server_utils import build_logger
from tool_server.tool_workers.offline_workers.base_offline_worker import BaseOfflineWorker
from tool_server.utils.error_codes import *

logger = build_logger("get_weather_worker")

class GetWeather(BaseOfflineWorker):
    """
    获取指定地点和时间的天气信息
    支持查询前后7天的天气
    """
    
    def __init__(self):
        super().__init__(model_name="GetWeather")
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Get weather information for a specific location and date. Supports queries for the past 7 days, today, and the next 7 days.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to query weather for, e.g., 'Beijing', 'New York', 'London'"
                        },
                        "date": {
                            "type": "string",
                            "description": "The date to query weather for in format 'YYYY-MM-DD', e.g., '2024-03-15'. If not provided, returns today's weather. Can query from 7 days ago to 7 days in the future."
                        },
                        "unit": {
                            "type": "string",
                            "description": "Temperature unit: 'celsius' or 'fahrenheit' (default: 'celsius')",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }
        
        # 天气状况选项
        self.weather_conditions = [
            "Sunny", "Partly Cloudy", "Cloudy", "Overcast",
            "Light Rain", "Rain", "Heavy Rain", "Thunderstorm",
            "Light Snow", "Snow", "Heavy Snow",
            "Foggy", "Windy", "Clear"
        ]
        
        # 风向选项
        self.wind_directions = [
            "N", "NE", "E", "SE", "S", "SW", "W", "NW"
        ]
        
        # 为了保持一致性，使用位置和日期作为随机种子
        self.random_seed_base = 42

    def _execute(self, params):
        """执行天气查询"""
        try:
            # 提取参数
            location = params["location"]
            date_str = params.get("date")
            unit = params.get("unit", "celsius")
            
            # 处理日期
            if date_str:
                try:
                    query_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    return {
                        "status": "failed",
                        "message": "Invalid date format. Please use 'YYYY-MM-DD' format.",
                        "error_code": INVALID_PARAMETERS
                    }
            else:
                query_date = datetime.now()
                date_str = query_date.strftime("%Y-%m-%d")
            
            # 检查日期范围（前后7天）
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            days_diff = (query_date - today).days
            
            if days_diff < -7 or days_diff > 7:
                return {
                    "status": "failed",
                    "message": "Date out of range. Can only query weather from 7 days ago to 7 days in the future.",
                    "error_code": INVALID_PARAMETERS
                }
            
            # 生成天气信息
            weather_info = self._generate_weather(location, date_str, unit)
            
            return {
                "status": "success",
                "location": location,
                "date": date_str,
                "weather": weather_info,
                "message": f"Weather information retrieved for {location} on {date_str}",
                "error_code": SUCCESS
            }
            
        except Exception as e:
            logger.error(f"Error getting weather: {str(e)}")
            return {
                "status": "failed",
                "message": f"Error getting weather: {str(e)}",
                "error_code": TOOL_RUN_FAILED
            }
    
    def _generate_weather(self, location, date_str, unit):
        """
        生成模拟的天气信息
        使用位置和日期作为随机种子以保持一致性
        """
        # 创建基于位置和日期的随机种子
        seed_str = f"{location.lower()}_{date_str}"
        seed = hash(seed_str) % (2**31)
        random.seed(seed)
        
        # 生成天气状况
        condition = random.choice(self.weather_conditions)
        
        # 根据天气状况生成合理的温度范围
        if "Snow" in condition:
            temp_min = random.randint(-10, 0)
            temp_max = temp_min + random.randint(5, 10)
        elif "Rain" in condition or condition == "Cloudy":
            temp_min = random.randint(10, 20)
            temp_max = temp_min + random.randint(5, 10)
        elif condition in ["Sunny", "Clear"]:
            temp_min = random.randint(15, 25)
            temp_max = temp_min + random.randint(8, 15)
        else:
            temp_min = random.randint(10, 20)
            temp_max = temp_min + random.randint(5, 12)
        
        # 如果使用华氏度，转换温度
        if unit == "fahrenheit":
            temp_min = int(temp_min * 9/5 + 32)
            temp_max = int(temp_max * 9/5 + 32)
            temp_unit = "°F"
        else:
            temp_unit = "°C"
        
        # 生成其他天气参数
        humidity = random.randint(30, 90)
        wind_speed = random.randint(5, 30)
        wind_direction = random.choice(self.wind_directions)
        
        # 根据天气状况调整降水概率
        if "Rain" in condition or "Snow" in condition or "Thunderstorm" in condition:
            precipitation_chance = random.randint(60, 95)
        elif "Cloudy" in condition:
            precipitation_chance = random.randint(20, 50)
        else:
            precipitation_chance = random.randint(0, 20)
        
        # 生成紫外线指数（0-11）
        if condition in ["Sunny", "Clear"]:
            uv_index = random.randint(6, 11)
        elif "Cloudy" in condition:
            uv_index = random.randint(3, 6)
        else:
            uv_index = random.randint(0, 4)
        
        # 生成空气质量指数（AQI）
        aqi = random.randint(20, 150)
        if aqi <= 50:
            aqi_category = "Good"
        elif aqi <= 100:
            aqi_category = "Moderate"
        elif aqi <= 150:
            aqi_category = "Unhealthy for Sensitive Groups"
        else:
            aqi_category = "Unhealthy"
        
        weather_info = {
            "condition": condition,
            "temperature": {
                "min": temp_min,
                "max": temp_max,
                "unit": temp_unit
            },
            "humidity": f"{humidity}%",
            "wind": {
                "speed": f"{wind_speed} km/h",
                "direction": wind_direction
            },
            "precipitation_chance": f"{precipitation_chance}%",
            "uv_index": uv_index,
            "air_quality": {
                "aqi": aqi,
                "category": aqi_category
            },
            "description": self._generate_description(condition, temp_min, temp_max, temp_unit)
        }
        
        return weather_info
    
    def _generate_description(self, condition, temp_min, temp_max, unit):
        """生成天气描述"""
        descriptions = {
            "Sunny": f"A beautiful sunny day with clear skies. Temperature ranging from {temp_min}{unit} to {temp_max}{unit}.",
            "Partly Cloudy": f"Partly cloudy skies throughout the day. Expect temperatures between {temp_min}{unit} and {temp_max}{unit}.",
            "Cloudy": f"Overcast conditions expected. Temperature will be around {temp_min}{unit} to {temp_max}{unit}.",
            "Overcast": f"Gray and overcast skies. Temperatures ranging from {temp_min}{unit} to {temp_max}{unit}.",
            "Light Rain": f"Light rain showers expected. Temperature between {temp_min}{unit} and {temp_max}{unit}. Don't forget your umbrella!",
            "Rain": f"Rainy conditions throughout the day. Temperatures from {temp_min}{unit} to {temp_max}{unit}. Umbrella recommended.",
            "Heavy Rain": f"Heavy rainfall expected. Stay indoors if possible. Temperature: {temp_min}{unit} to {temp_max}{unit}.",
            "Thunderstorm": f"Thunderstorms likely. Temperature ranging from {temp_min}{unit} to {temp_max}{unit}. Stay safe indoors.",
            "Light Snow": f"Light snowfall expected. Temperature between {temp_min}{unit} and {temp_max}{unit}. Drive carefully.",
            "Snow": f"Snow throughout the day. Temperatures from {temp_min}{unit} to {temp_max}{unit}. Dress warmly.",
            "Heavy Snow": f"Heavy snowfall expected. Temperature: {temp_min}{unit} to {temp_max}{unit}. Avoid travel if possible.",
            "Foggy": f"Foggy conditions expected. Limited visibility. Temperature between {temp_min}{unit} and {temp_max}{unit}.",
            "Windy": f"Windy conditions throughout the day. Temperature ranging from {temp_min}{unit} to {temp_max}{unit}.",
            "Clear": f"Clear skies and pleasant weather. Temperature from {temp_min}{unit} to {temp_max}{unit}."
        }
        
        return descriptions.get(condition, f"Weather condition: {condition}. Temperature: {temp_min}{unit} to {temp_max}{unit}.")
    
    def verify_tool_parameter(self, params):
        """验证工具的输入参数"""
        try:
            # 检查location是否存在
            if "location" not in params:
                raise ValueError("Missing required parameter: location")
            
            location = params["location"]
            if not isinstance(location, str) or len(location.strip()) == 0:
                raise ValueError("location must be a non-empty string")
            
            # 验证date参数（如果提供）
            date_str = params.get("date")
            if date_str:
                if not isinstance(date_str, str):
                    raise ValueError("date must be a string in format 'YYYY-MM-DD'")
                
                # 验证日期格式
                try:
                    query_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    raise ValueError("Invalid date format. Please use 'YYYY-MM-DD' format, e.g., '2024-03-15'")
                
                # 检查日期范围
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                days_diff = (query_date - today).days
                
                if days_diff < -7:
                    raise ValueError("Cannot query weather from more than 7 days ago")
                if days_diff > 7:
                    raise ValueError("Cannot query weather for more than 7 days in the future")
            
            # 验证unit参数（如果提供）
            unit = params.get("unit", "celsius")
            if isinstance(unit, str):
                unit = unit.lower()
            
            if unit not in ["celsius", "fahrenheit"]:
                raise ValueError("unit must be either 'celsius' or 'fahrenheit'")
            
            # 创建新的参数字典
            new_params = {
                "location": location.strip(),
                "unit": unit
            }
            
            if date_str:
                new_params["date"] = date_str
            
            return {
                "params_qualified_reward": 1,
                "params_qualified": True,
                "new_params": new_params
            }
            
        except Exception as e:
            error_info = str(e)
            return {
                "params_qualified_reward": 0,
                "params_qualified": False,
                "error_info": error_info,
                "new_params": None
            }