# pccip install pymysql sqlacodegen
import os
import subprocess
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 创建数据库引擎
# 与 endpoints.yml database.url 对齐，支持从环境变量覆盖
db_host = os.getenv("MYSQL_HOST", "52.231.65.205")
db_port = int(os.getenv("MYSQL_PORT", "3306"))
db_name = os.getenv("MYSQL_DB", "ecs")
db_user_name = os.getenv("MYSQL_USER", "root")
db_password = os.getenv("MYSQL_PASSWORD", "123456")
url = f"mysql+pymysql://{db_user_name}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8"

# 配置会话工厂
engine = create_engine(url)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


if __name__ == "__main__":

    def export_db_table_class(run=False):
        """将数据库表映射为Python类"""
        if not run:
            return
        output_path = "db_table_class.py"

        cmd = ["python", "-m", "sqlacodegen", url]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)

    export_db_table_class(True)
