# 模糊弦号处理
clarity="blurry"时：直接用弦号调lookup_by_hull_number，未命中则retrieve_by_description兜底。不做二次识别，尽力查找。