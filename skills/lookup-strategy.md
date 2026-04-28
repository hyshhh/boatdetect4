# 查找策略
1. 有弦号→lookup_by_hull_number，命中即结束
2. 未命中或无弦号→retrieve_by_description语义检索(top_k=3)
禁止跳步，禁止同时调用多个工具