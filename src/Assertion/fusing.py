"""
Kết hợp kết quả của rule based và bge-m3.
Nếu assertion sentence = "[ENTITY]." -> assertion bỏ trống.
Ưu tiên lấy kết quả của rule based (nếu không null).
Thêm trường "assertion_result".
"""

