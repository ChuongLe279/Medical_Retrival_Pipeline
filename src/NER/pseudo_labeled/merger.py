"""
merge vistral và hardcode lại
100 vistral + 100 rule-based -> 100 merge output file

Chiến thuật:
1. Gộp 2 list entity lại, sort theo name. Output tên 1.json, 2.json, 3.json, vv.
2. Duyệt qua, nhóm các entity có OVERLAP vị trí (kể cả overlap 1 phần) thành 1 "cluster"
3. Với mỗi cluster:
   a. Nếu chỉ có 1 entity (không ai overlap) → giữ nguyên, thêm vào kết quả. Ghi "source":"(dựa trên source của entity)"
   b. Nếu nhiều entity overlap nhau:
      - Cùng type, cùng span → merge làm 1, giữ lại (không trùng lặp). Ghi "source":"both"
      - Cùng type, span khác nhau (1 nguồn bắt dài hơn/ngắn hơn) 
        → ưu tiên span của nguồn có độ tin cậy cao hơn cho type đó (bảng mục 2)
      - Khác type (rule nói THUỐC, LLM nói CHẨN_ĐOÁN cho cùng vị trí)
        → ưu tiên theo bảng type ở mục 2, log lại case này để review thủ công
4. Với vùng text KHÔNG có rule-based bắt được nhưng Vistral có
   → giữ nguyên kết quả Vistral (rule-based vốn không kỳ vọng bao phủ hết, chỉ mạnh ở vùng có pattern)
"""
