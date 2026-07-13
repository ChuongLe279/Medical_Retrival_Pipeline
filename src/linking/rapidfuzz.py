"""
Không thể tin tưởng hoàn toàn vào kết quả của Sapbert (VD: 2 mã khác nhau có độ chính xác đều bằng 1.0)
Ta kết hợp kết quả của sapbert và rapidfuzz.
Rapid fuzz không cần embed. So sánh thẳng query với text list luôn.
"""

from rapidfuzz import process, fuzz
