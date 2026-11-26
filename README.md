# FCVG101 — Smart Organic Farm Assistant

A reproducible computer-vision project reference implementation. This package includes actual **controlled synthetic Task 1 experiments**; it does not include a trained fruit model or measured Task 2 performance yet. No synthetic image is passed off as a real fruit photograph. Do not submit the draft unchanged.

## 1. Mở project bằng VS Code (Windows)

1. Giải nén ZIP. Mở VS Code → File → Open Folder → chọn thư mục `FCVG101_Fruit_Inspection` (có file `app.py` và `requirements.txt`).
2. Terminal → New Terminal. Khuyến nghị Python 3.11 hoặc 3.12, dùng một môi trường ảo mới để tránh xung đột.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu bạn đã có `.venv` tương thích thì không cần tạo lại. Các lệnh bên dưới dùng thẳng Python trong môi trường ảo, không cần thay đổi chính sách PowerShell. Linux/macOS: thay `.\.venv\Scripts\python.exe` bằng `./.venv/bin/python`.

## 2. Task 1 — classical computer vision

Chạy thử:

```powershell
.\.venv\Scripts\python.exe -m src.experiment
```

Kết quả tạo trong `outputs/task1/`: ảnh filtering, edges, frequency, features, ORB/SIFT matching, calibration, PSNR chart, `results.json` và `calibration.npz`. Mặc định sử dụng hình minh họa trái cây và checkerboard mô phỏng; đây là kiểm thử kỹ thuật, không phải ảnh thực nghiệm ngoài đời.

**Để lấy bằng chứng thật:** đặt hai ảnh tự chụp của cùng trái cây vào `data/view1.jpg` và `data/view2.jpg`; chụp 10–15 ảnh checkerboard từ cùng camera, độ phân giải và nhiều góc nghiêng, lưu trong `data/checkerboards/`. Dùng checkerboard 10 × 7 ô (9 × 6 góc trong), đo kích thước ô thực tế bằng mm. Nếu ô 25 mm:

```powershell
.\.venv\Scripts\python.exe -m src.experiment --image data/view1.jpg --second data/view2.jpg --checkerboards data/checkerboards --cols 9 --rows 6 --square-mm 25
```

Nếu board khác kích thước, thay các giá trị tương ứng. Tối thiểu 5 ảnh checkerboard được nhận diện, nhiều góc và vị trí đa dạng tốt hơn. Không dùng ma trận K mô phỏng cho camera thật.

## 3. Task 2 — dataset và model

Dataset: [Sultana, Jahan & Uddin (2022), Mendeley Data](https://data.mendeley.com/datasets/bdd69gyhv8/1), CC BY 4.0; [Hugging Face mirror](https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification). Raw subset: 3,200 images, 8 fruit types, 2 classes, 1,600 each (as published). The download script requires Internet. It does not use pre-augmented samples.

```powershell
.\.venv\Scripts\python.exe -m src.download_data
.\.venv\Scripts\python.exe -m src.train --epochs 12
```

Không đóng terminal khi tải hoặc train. Model chọn theo validation accuracy rồi đánh giá test riêng. Nếu máy yếu, chạy `--epochs 1` để kiểm tra pipeline trước, sau đó chạy lại 12 epochs để lấy kết quả chính thức. Thời gian và kết quả phụ thuộc máy và dữ liệu. `data/raw/manifest.csv` lưu `file,label,fruit`. Train làm sạch ảnh hỏng và trùng, split theo loại và chất lượng, augmentation chỉ ở train. `outputs/task2/` chứa biểu đồ, JSON metrics, history CSV, splits CSV. `models/fruit_cnn.pt` là model tốt nhất. Kiểm tra log và biểu đồ; không dùng số liệu chưa thực sự tạo.

## 4. Task 3 — ứng dụng

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Mở địa chỉ Local URL trong terminal (thường `http://localhost:8501`). Chọn mode ở sidebar. Nếu chưa train, mục Classify fruit báo thiếu model. Chụp ảnh màn hình từng mode và lưu vào `report/screenshots/` để bổ sung báo cáo. App chưa được browser-test trong môi trường tạo bộ file này.

## 5. GitHub và report

- Tạo repository của bạn tại https://github.com/new (đặt tên `FCVG101-Fruit-Inspection`). Chọn public nếu giảng viên yêu cầu và bạn được phép; không đưa dữ liệu cá nhân lên repository.
- Cách không cần Git terminal: trong repository chọn Add file → Upload files, kéo các thư mục `src`, `tests`, `report` (source script), và các file `app.py`, `README.md`, `requirements.txt`, `.gitignore`; giữ nguyên cấu trúc. Không tải `.venv` hoặc dataset lên. Dùng branch `main` để các đường dẫn code trong báo cáo đúng. Kiểm tra file trên GitHub đã mở được.
- Model: sau khi train, tạo GitHub Release rồi tải `models/fruit_cnn.pt` lên mục assets, hoặc dùng nơi lưu trữ được trường cho phép. Sao chép đường dẫn truy cập thực tế và thử mở bằng chế độ không đăng nhập theo yêu cầu giảng viên.
- Đề bắt buộc PDF có code selectable as text. Nội dung chính dùng link GitHub theo mong muốn của bạn; thêm `--include-code` để đáp ứng yêu cầu appendix, trừ khi giảng viên xác nhận chỉ cần link.

Ví dụ (thay URL mẫu bằng URL thật, không sử dụng nguyên mẫu):

```powershell
.\.venv\Scripts\python.exe report/build_report.py --repo "https://github.com/YOUR-USERNAME/FCVG101-Fruit-Inspection" --model-link "YOUR-ACTUAL-MODEL-LINK" --student "Your Name" --student-id "Your ID" --include-code
```

File PDF được tạo ở `report/`. Nếu còn thiếu dữ liệu hoặc link, file mang tên `FCVG101_Report_Draft.pdf`. Khi có các kết quả thật, link, và appendix, file có tên `FCVG101_Report.pdf`. Dù vậy, vẫn phải điền thông tin cá nhân, kiểm tra ảnh chụp app, giải thích kết quả thật và xác minh quyền truy cập. Dùng file PDF dưới 5 MB để nộp CMS. Không khai báo kết quả mô phỏng là ảnh chụp thật. Đọc chính sách học thuật và ghi nhận nguồn / trợ giúp AI theo quy định của trường.

## 6. Kiểm tra code

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 7. Rubric evidence map

| Criterion | Points | File and evidence | Status in supplied package |
|---|---:|---|---|
| 1.1 Camera geometry/calibration | 5 | `src/vision.py`, `outputs/task1/calibration.jpg`, `results.json` | Simulated calibration executed; real camera pending |
| 1.2 Filters/edges | 5 | `outputs/task1/filtering.jpg`, `edges.jpg`, `frequency.jpg`, `psnr.png` | Executed on controlled synthetic input |
| 1.3 Features/matching | 5 | `features.jpg`, `orb_matches.jpg`, `sift_matches.jpg` | Executed on synthetic view; physical second view pending |
| 1.4 Code quality | 5 | `src/vision.py`, `src/experiment.py`, tests | Unit tests passed |
| 2.1 Data pipeline | 15 | `src/download_data.py`, `src/train.py`, manifest/splits | Code written; download and real-data execution pending |
| 2.2 CNN | 15 | `src/model.py`, `src/train.py`, model checkpoint | Forward pass tested; training pending |
| 2.3 Evaluation | 5 | `outputs/task2/metrics.json`, plots | No real-data metrics yet |
| 3.1 UI logic | 15 | `app.py` | Code compiled; browser test pending |
| 3.2 Integration | 15 | `app.py`, shared vision/model code | Model and manual app evidence pending |
| 4.1 Report | 10 | `report/FCVG101_Report_Draft.pdf`, report generator | Draft with measured simulation; finalize after real experiments |
| 4.2 README | 5 | This file, modular source | Included |
