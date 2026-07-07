# Quy tắc pull request

- Khóa nhánh main. Không ai được push trực tiếp.
- Review PR phải theo thứ tự lần lượt để đảm bảo không conflict, nếu có conflict thì phải resolve
- Sẽ có 2 nhánh chính, 1 nhánh main là bản final, 1 nhánh dev do mình quản lí
- Nhánh dev để merge các pull request thay đổi, fix, hoặc cherry pick commit, ...
- Nhánh main chỉ pull rebase dev về hoặc cherry pick commit từ dev thôi để luôn giữ ổn định
- Pull nhánh dev và tên gì thì tuỳ nhưng khi pull request bắt buộc phải đặt tên theo chuẩn sau:
Ví dụ:
```
feat/save-board
fix/login-bug
```
- Ví dụ khi BE làm chức năng lưu, tạo nhánh feat/save-board. Xong thì tạo Pull Request ghép vào dev. Phải có ít nhất 1 người khác vào đọc code và bấm Approve mới được merge.
- Khi merge xong thì xoá nhánh feature/save-board

Ví dụ về format pull request:
- Lưu ý: chỗ nào có thay dổi thì viết, còn không có thay đổi gì ở đấy thì không ghi. Ví dụ không fix gì thì bỏ cái Fix đi

---
# Tên tính năng (ví dụ: Auth và lưu bản vẽ)
- Mô tả nhanh về thay đổi

# Các thay đổi
## Tính năng
- Thêm API Save/Load bản vẽ
- Thêm API Đăng ký / Đăng nhập

## Fix
- Fix không hiển thị hình ảnh trên màn phân giải cao

## Khác
- Cập nhật phiên bản lên 0.1.14

Chi tiết thêm (nếu có) ...
