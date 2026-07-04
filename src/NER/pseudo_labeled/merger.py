"""
Chiến lược merge output của vistral và rule based
1. Nếu không overlap:
   giữ nguyên entity.

2. Nếu overlap cùng type + cùng span:
   merge thành 1 entity, source = "both".

3. Nếu overlap cùng type nhưng khác span:
   chọn theo TYPE_PRIORITY[type].
   Riêng TÊN_XÉT_NGHIỆM:
     - nếu một span là viết tắt chuẩn như ECG/WBC/CRP/HbA1c → rule tốt
     - nếu Vistral bắt tên đầy đủ hơn → cân nhắc chọn span dài hơn
   log lại nếu span lệch nhiều.

4. Nếu overlap khác type:
   chọn entity có source ưu tiên cao hơn theo type của chính entity đó.
   Luôn log conflict này để review thủ công.

5. Nếu source = section_rule:
   chỉ dùng khi không có rule hoặc vistral overlap tốt hơn.
"""
