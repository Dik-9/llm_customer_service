# -*- coding: utf-8 -*-
"""
执行 ecs.sql 建库建表 + 灌入基础数据。

ecs.sql 第 1-3 行已包含 drop database if exists ecs; create database ecs; use ecs;
所以直接读取文件按分号分割执行即可。
"""
import os
import re
import sys
from pathlib import Path

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import pymysql
import pymysql.cursors

DB_HOST = os.getenv("MYSQL_HOST", "52.231.65.205")
DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456")

SQL_FILE = r"E:\百度网盘文件\07-项目_智能客服系统\02_资料\业务数据准备\ecs.sql"


def split_sql_statements(sql_text: str) -> list:
    """按分号分割 SQL 语句，跳过注释行。

    注意：这个简单分割器假设 SQL 里没有包含分号的字符串字面量
    （ecs.sql 是建表脚本，没有这种边缘情况）。
    """
    statements = []
    current = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        # 跳过空行和注释行
        if not stripped or stripped.startswith("--") or stripped.startswith("#"):
            continue
        current.append(line)
        # 行尾或行内有分号 → 一条语句结束
        if ";" in line:
            stmt = "\n".join(current).strip()
            # 可能一行有多条语句（用 ; 分隔），再切一次
            for s in stmt.split(";"):
                s = s.strip()
                if s:
                    statements.append(s)
            current = []
    # 处理末尾未以分号结尾的内容（应该没有）
    if current:
        stmt = "\n".join(current).strip()
        if stmt:
            statements.append(stmt)
    return statements


def main():
    if not Path(SQL_FILE).exists():
        print(f"❌ 找不到 SQL 文件: {SQL_FILE}")
        sys.exit(1)

    print(f"读取 SQL 文件: {SQL_FILE}")
    with open(SQL_FILE, "r", encoding="utf-8") as f:
        sql_text = f.read()
    print(f"文件大小: {len(sql_text)} 字节")

    statements = split_sql_statements(sql_text)
    print(f"分割出 {len(statements)} 条 SQL 语句")
    print()

    # 先连 MySQL（不指定库），执行 drop database / create database / use ecs
    print(f"连接 MySQL: {DB_USER}@{DB_HOST}:{DB_PORT}")
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

    cur = conn.cursor()

    # 执行每条语句
    success_count = 0
    error_count = 0
    current_db = None

    for i, stmt in enumerate(statements, 1):
        # 取语句前 60 字符做日志
        preview = re.sub(r"\s+", " ", stmt)[:60]
        try:
            # 跟踪当前 use 的库
            upper = stmt.upper()
            if upper.startswith("USE "):
                current_db = stmt[4:].strip().strip("`")
                print(f"[{i:3d}] USE → {current_db}")
                cur.execute(stmt)
                success_count += 1
            elif upper.startswith("DROP DATABASE"):
                print(f"[{i:3d}] DROP DATABASE → {preview}")
                cur.execute(stmt)
                success_count += 1
            elif upper.startswith("CREATE DATABASE"):
                print(f"[{i:3d}] CREATE DATABASE → {preview}")
                cur.execute(stmt)
                success_count += 1
            else:
                # 普通建表/insert 语句，简洁日志
                first_kw = upper.split()[0] if upper.split() else "?"
                if first_kw == "CREATE":
                    # 提取表名
                    m = re.search(r"CREATE\s+TABLE\s+(\w+)", stmt, re.IGNORECASE)
                    if m:
                        print(f"[{i:3d}] CREATE TABLE {m.group(1)}")
                    else:
                        print(f"[{i:3d}] CREATE ...")
                elif first_kw == "INSERT":
                    m = re.search(r"INSERT\s+INTO\s+(\w+)", stmt, re.IGNORECASE)
                    if m:
                        print(f"[{i:3d}] INSERT INTO {m.group(1)} ({len(stmt)} chars)")
                    else:
                        print(f"[{i:3d}] INSERT ...")
                else:
                    print(f"[{i:3d}] {first_kw} ...")
                cur.execute(stmt)
                success_count += 1
        except pymysql.Error as e:
            error_count += 1
            print(f"[{i:3d}] ❌ 错误: {e}")
            print(f"      语句前 200 字符: {stmt[:200]}")

    print()
    print(f"=== 执行完成 ===")
    print(f"成功: {success_count} 条")
    print(f"失败: {error_count} 条")

    # 验证 ecs 库的表
    print()
    print("=== 验证 ecs 库的表 ===")
    cur.execute("USE ecs")
    cur.execute("SHOW TABLES")
    tables = [row[list(row.keys())[0]] for row in cur.fetchall()]
    print(f"ecs 库共 {len(tables)} 张表:")
    for t in tables:
        cur.execute(f"SELECT COUNT(*) AS c FROM `{t}`")
        cnt = cur.fetchone()["c"]
        print(f"  - {t:35s}  {cnt} 行")

    conn.close()
    print()
    print("✅ 完成。下一步：修改 gen_data.py 后跑 gen_data.py 灌模拟数据")


if __name__ == "__main__":
    main()
