import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

def fetch_ulb_fraud_data() -> pd.DataFrame:
    """
    使用 Google Cloud Client Library 从 BigQuery 中提取布鲁塞尔自由大学(ULB)信用卡欺诈样本数据。
    返回:
        pd.DataFrame: 包含风控特征、金额和Class标签的 Pandas DataFrame
    """
    # 1. 初始化 BigQuery 客户端（它会自动识别你在 GCP 笔记本中的托管身份凭证）
    client = bigquery.Client()
    
    # 2. 编写标准 SQL 查询语句
    # 注意：为了代码的安全性和规范性，库名、表名最好用反引号包裹
    query = """
        SELECT * 
        FROM `fraud_detection.ulb_credit_card_fraud`
        # LIMIT 1000
    """
    
    print("⏳ 正在建立安全连接并从 BigQuery 提取 ULB 数据...")
    
    try:
        # 3. 执行查询并利用官方集成的 .to_dataframe() 直接转化为 Pandas DataFrame
        query_job = client.query(query)
        df = query_job.to_dataframe()
        
        print(f"✅ 数据提取成功！共成功加载 {len(df)} 条风控样本数据。")
        return df
        
    except GoogleCloudError as e:
        print(f"❌ GCP 基础架构连接或查询失败: {e}")
        raise
    except Exception as e:
        print(f"❌ 发生未预期的本地数据转换错误: {e}")
        raise

if __name__ == "__main__":
    # 本地直接运行此脚本时的自测逻辑
    fraud_df = fetch_ulb_fraud_data()
    print("\n🔍 预览前 2 条布鲁塞尔自由大学风控特征数据:")
    print(fraud_df.head(2))
