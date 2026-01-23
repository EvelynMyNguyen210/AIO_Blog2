# 1. Việc phát triển AI chatbot ngày nay trở nên dễ tiếp cận hơn bao giờ hết

# 1.1. Sự thay đổi trong cách tiếp cận phát triển chatbot

Khi nghĩ đến việc xây dựng một AI chatbot, nhiều người thường có ấn tượng rằng đây là một công việc phức tạp, đòi hỏi kiến thức chuyên sâu về trí tuệ nhân tạo và máy học. Tuy nhiên, trong những năm gần đây, bối cảnh phát triển chatbot đã thay đổi đáng kể (OpenAI, 2025; Rasiksuhail, 2026).

Sự thay đổi này đến từ hai yếu tố chính:

**Thứ nhất, sự xuất hiện của các dịch vụ AI dưới dạng API:** Các tổ chức như OpenAI, Anthropic, và Google đã đầu tư hàng tỷ USD để phát triển các mô hình ngôn ngữ lớn (Large Language Models - LLMs) cực kỳ mạnh mẽ. Thay vì yêu cầu người dùng tự xây dựng và huấn luyện mô hình, các công ty này cung cấp khả năng truy cập vào các mô hình này thông qua giao diện lập trình ứng dụng (Application Programming Interface - API) (OpenAI, 2025; Anthropic, 2025; Google AI for Developers, 2026).

**Thứ hai, sự phát triển của các công cụ và thư viện hỗ trợ.** Ngày nay, việc tích hợp AI vào ứng dụng đã trở nên đơn giản hơn nhiều nhờ các framework và thư viện được tối ưu hóa. Điều này cho phép các nhà phát triển tập trung vào logic nghiệp vụ thay vì phải lo lắng về các chi tiết kỹ thuật phức tạp của machine learning.

## 1.2. Phân biệt giữa phát triển mô hình AI và xây dựng chatbot
Một điểm quan trọng cần làm rõ là sự khác biệt giữa phát triển mô hình AI từ đầu và xây dựng một ứng dụng chatbot (Hire A.I. Developers, 2025).

**Phát triển mô hình AI đòi hỏi:**

- Kiến thức chuyên sâu về neural networks, deep learning architectures
- Khả năng xử lý và chuẩn bị dữ liệu huấn luyện quy mô lớn
- Tài nguyên tính toán mạnh mẽ (GPU clusters, TPU)
- Thời gian và chi phí đáng kể cho quá trình huấn luyện

**Ngược lại, xây dựng một chatbot tập trung vào:**

- Thiết kế luồng hội thoại
- Tích hợp các thành phần hệ thống
- Quản lý ngữ cảnh và trạng thái
- Xử lý logic nghiệp vụ cụ thể

<p align="center">
  <img src="images\blog2_so_sanh_quy_trinh.png" style="margin: 0 auto; display: block;"><br/>
  <em>Hình 1.1. So sánh quy trình "Phát triển mô hình AI từ đầu" và "Xây dựng chatbot sử dụng AI API"</em>
</p>

Có thể hình dung quá trình xây dựng chatbot giống như việc xây dựng một tòa nhà:

Khi xây nhà, người ta không cần phải tự sản xuất gạch, xi măng từ nguyên liệu thô. Thay vào đó, kỹ sư và kiến trúc sư tập trung vào việc thiết kế bản vẽ, lựa chọn vật liệu phù hợp, và tổ chức thi công sao cho hiệu quả.

Tương tự, khi xây dựng chatbot:

- Mô hình AI giống như vật liệu xây dựng đã được sản xuất sẵn
- Công việc của nhà phát triển là thiết kế kiến trúc hệ thống
- Kết nối các thành phần để tạo ra sản phẩm hoàn chỉnh

# 2. Một AI chatbot tối thiểu gồm những thành phần nào

Một hệ thống chatbot dựa trên AI không chỉ đơn thuần là một mô hình ngôn ngữ lớn, mà là một hệ thống tích hợp nhiều thành phần phối hợp chặt chẽ để tạo ra trải nghiệm hội thoại tự nhiên và hiệu quả (ScienceDirect, 2025). Dưới đây là bốn thành phần cốt lõi cần thiết trong một AI chatbot tối thiểu:

<p align="center">
  <img src="images\blog_2_thanh_phan_he_thong_AI_chatbot.png" style="margin: 0 auto; display: block;"><br/>
  <em>Hình 2.1. Các thành phần chính của một hệ thống AI chatbot</em>
</p>

## 2.1. Giao diện người dùng: 
**Vai trò:** Tạo điểm tiếp xúc giữa người dùng và hệ thống chatbot

Giao diện người dùng là lớp trực tiếp tương tác với người dùng cuối. Tuỳ thuộc vào platform và use case, giao diện có thể được triển khai dưới nhiều hình thức.

**Các dạng giao diện phổ biến:**

- Web-based interface: Widget chat tích hợp trên website doanh nghiệp
- Mobile application: Giao diện chat trong ứng dụng di động
- Messaging platforms: Tích hợp vào Facebook Messenger, Telegram, Zalo
- Voice interface: Giao diện giọng nói như Siri, Google Assistant
- Command-line interface: Giao diện dòng lệnh cho mục đích testing và development

**Chức năng chính:**

- Thu thập và chuẩn hóa input từ người dùng (văn bản, giọng nói)
- Hiển thị response dưới định dạng phù hợp

## 2.2. Lớp xử lý logic: 

**Vai trò:** Trung tâm điều phối và xử lý luồng hội thoại
Đây là thành phần mà nhà phát triển sẽ triển khai chủ yếu, đảm nhiệm việc điều phối toàn bộ quy trình từ khi nhận input đến khi trả về response.

### a) Tiền xử lý dữ liệu đầu vào
Trước khi gửi dữ liệu đến mô hình AI, cần thực hiện các bước chuẩn hóa:

- Loại bỏ các ký tự không cần thiết (khoảng trắng thừa, kí tự đặc biệt)
- Chuẩn hóa định dạng văn bản
- Xử lý lỗi chính tả cơ bản
- Phát hiện ngôn ngữ cho hệ thống đa ngôn ngữ

### b) Quản lý ngữ cảnh hội thoại
Một chatbot chất lượng cần có khả năng duy trì ngữ cảnh xuyên suốt cuộc hội thoại:

- Lưu trữ lịch sử trao đổi
- Theo dõi trạng thái hiện tại của hội thoại
- Quản lý session riêng biệt cho từng người dùng

### c) Định tuyến logic
Quyết định cách xử lý phù hợp cho từng loại request:

- Xác định xem câu hỏi nên được xử lý dựa trên quy tắc hay dựa trên AI
- Kích hoạt các chức năng đặc biệt (truy vấn cơ sở dữ liệu, lấy dữ liệu từ bên ngoài)
- Xử lý các lệnh hệ thống

### d) Hậu xử lý kết quả
Trước khi trả về cho người dùng, phản hồi cần được xử lý:

- Định dạng văn bản
- Kiểm tra độ dài phù hợp với platform
- Lọc nội dung không phù hợp

## 2.3. Mô hình AI hoặc dịch vụ truy cập: 

**Vai trò:** Xử lý ngôn ngữ tự nhiên và sinh câu trả lời

Đây là thành phần cung cấp khả năng hiểu và sinh ngôn ngữ tự nhiên cho chatbot. Có hai phương pháp chính để tích hợp thành phần này:

#### **Phương pháp 1: Sử dụng AI thông qua API**
Đây là phương pháp phổ biến nhất trong các ứng dụng thực tế. Nhà phát triển sử dụng các mô hình đã được huấn luyện sẵn thông qua API (OpenAI, 2025; Google AI for Developers, 2026).



| Nhà cung cấp              | Mô hình mới nhất (tháng 1/2026)                  | Điểm mạnh                                                                 | Hạn chế                                                                 |
|---------------------------|--------------------------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **OpenAI**               | GPT-5.2 (và biến thể Pro, Instant), gpt-oss (open-weight 120B/20B) | Hiệu năng reasoning dẫn đầu nhiều benchmark, hỗ trợ đa phương thức mạnh (text, hình ảnh, giọng nói), context window lên đến 400K tokens, tích hợp tốt cho enterprise và AI agents | Chi phí cao khi sử dụng lớn, phụ thuộc hoàn toàn vào hạ tầng OpenAI, một số lo ngại về quyền riêng tư dữ liệu |
| **Anthropic**            | Claude Opus 4.5 (cùng Sonnet 4.5, Haiku 4.5)    | An toàn cao (constitutional AI), ngữ cảnh dài (~200K tokens), xuất sắc trong coding, AI agents và ứng dụng chuyên ngành (y tế, pháp lý), giảm thiểu hallucination hiệu quả | Tốc độ API đôi khi chậm hơn so với đối thủ, hỗ trợ đa phương thức còn hạn chế (chủ yếu tập trung vào text), chi phí cao cho model flagship |
| **Google**               | Gemini 3 Pro / Gemini 3 Flash (với Deep Think mode) | Ngữ cảnh cực dài (lên đến 1M tokens), đa phương thức toàn diện (text, hình ảnh, video, audio), tích hợp sâu với hệ sinh thái Google (Search, Workspace, YouTube), tốc độ cao ở biến thể Flash | Chi phí cao cho usage lớn, hệ sinh thái khép kín, phụ thuộc Google Cloud, một số tính năng vẫn ở giai đoạn experimental |
| **Hugging Face (Open-source hub)** | Llama series (Meta), Mistral/Mixtral (Mistral AI), Qwen, Gemma… | Miễn phí, mã nguồn mở, dễ tùy chỉnh và fine-tune, cộng đồng hỗ trợ lớn, có thể triển khai cục bộ hoặc offline, không phụ thuộc nhà cung cấp | Yêu cầu hạ tầng tính toán mạnh (GPU/server) để chạy hiệu quả, thiếu hỗ trợ chính thức và cập nhật tự động, hiệu năng có thể kém hơn so với các mô hình frontier closed ở một số nhiệm vụ phức tạp |

<p align="center">
  <em>Bảng 2.1 So sánh các nhà cung cấp mô hình ngôn ngữ lớn phổ biến (tháng 1/2026)</em>
</p>

**Ghi chú**:
- Bảng tập trung vào các mô hình cao cấp nhất và phổ biến nhất dùng để phát triển AI chatbot qua API hoặc tùy chỉnh cục bộ.
- Context window, chi phí và hiệu năng có thể thay đổi theo thời gian; khuyến nghị kiểm tra tài liệu chính thức của từng nhà cung cấp trước khi triển khai.
- Đối với dự án cá nhân hoặc học thuật, các mô hình mã nguồn mở trên Hugging Face thường là lựa chọn cân bằng giữa chi phí và khả năng tùy chỉnh.

**Ưu điểm của phương pháp API:**

- Không yêu cầu hạ tầng phức tạp (như cụm GPU lớn)
- Triển khai nhanh chóng, dễ dàng mở rộng quy mô
- Mô hình được nhà cung cấp cập nhật và cải tiến liên tục
- Có tài liệu hướng dẫn chi tiết và hỗ trợ kỹ thuật tốt

**Nhược điểm:**

- Chi phí vận hành tính theo mức độ sử dụng
- Phụ thuộc hoàn toàn vào nhà cung cấp bên thứ ba
- Độ trễ phản hồi có thể tăng do giao tiếp qua mạng 
- Khả năng tùy chỉnh sâu bị hạn chế

#### **Phương pháp 2: Tự triển khai và huấn luyện mô hình**
Phương pháp này phù hợp với các tổ chức có nhu cầu đặc biệt về bảo mật, tùy chỉnh hoặc chi phí.

Quy trình tổng quát:

1. Lựa chọn mô hình cơ sở: Chọn một mô hình ngôn ngữ lớn đã được huấn luyện sẵn phù hợp, ví dụ: BERT, GPT-2, LLaMA, Mistral, hoặc các biến thể mới hơn từ cộng đồng mã nguồn mở.
2. Chuẩn bị dữ liệu huấn luyện: Thu thập và xử lý dữ liệu chuyên biệt theo lĩnh vực (domain-specific), bao gồm việc gán nhãn (labeling) nếu cần, làm sạch dữ liệu và định dạng phù hợp với yêu cầu fine-tuning.
3. Thực hiện fine-tuning: Huấn luyện lại mô hình trên tập dữ liệu riêng, thường sử dụng các kỹ thuật tiết kiệm tài nguyên như PEFT (Parameter-Efficient Fine-Tuning), LoRA hoặc QLoRA để giảm yêu cầu phần cứng.
4. Triển khai mô hình: Đưa mô hình đã fine-tune lên máy chủ cục bộ, dịch vụ đám mây (cloud) hoặc nền tảng chuyên dụng để sẵn sàng phục vụ yêu cầu.
5. Xây dựng lớp API wrapper: Tạo một lớp giao tiếp (API layer) để các ứng dụng khác có thể dễ dàng gọi mô hình thông qua các endpoint chuẩn (ví dụ: REST API hoặc FastAPI).

**Ưu điểm:**

- Kiểm soát hoàn toàn dữ liệu và bảo mật thông tin 
- Có khả năng tùy chỉnh sâu, phù hợp tối ưu với lĩnh vực hoặc nhiệm vụ cụ thể
- Không phụ thuộc vào nhà cung cấp bên thứ ba
- Chi phí vận hành có thể thấp hơn đáng kể khi triển khai ở quy mô lớn và dài hạn

**Nhược điểm:**

- Yêu cầu trình độ chuyên môn cao về học máy và học sâu (ML/DL)
- Cần đầu tư hạ tầng tính toán mạnh mẽ (đặc biệt là GPU hoặc TPU)
- Thời gian phát triển và thử nghiệm kéo dài hơn nhiều so với phương pháp API
- Phải tự chịu trách nhiệm bảo trì, cập nhật và tối ưu hóa mô hình liên tục

## 2.4. Thành phần 4: Cơ sở tri thức - Tùy chọn
**Vai trò:** Cung cấp thông tin domain-specific mà mô hình AI tổng quát không có
Mặc dù các mô hình ngôn ngữ lớn đã được huấn luyện trên lượng dữ liệu khổng lồ, chúng không thể biết về:

- Thông tin nội bộ của tổ chức (giá sản phẩm, chính sách riêng)
- Dữ liệu thời gian thực (số lượng sản phẩm, lịch hẹn)
- Thông tin được cập nhật sau thời điểm training

**Lợi ích:**

- Chatbot có thể trả lời về thông tin nội bộ mà không cần tinh chỉnh mô hình
- Dễ dàng cập nhật cơ sở tri thức mà không cần huấn luyện lại mô hình
- Giảm AI bịa đặt thông tin
- Có thể truy vết nguồn gốc thông tin


# 3. Roadmap: 3 cấp độ xây dựng chatbot

## Level 1: Rule-based Chatbot

Chatbot rule-based hoạt động dựa trên các quy tắc được lập trình sẵn. Khi người dung đặt câu hỏi, chatbot sẽ xử lí và so sánh câu hỏi với các điều kiện có sẵn để đưa ra câu trả lời chính xác.

<p align="center">
  <img src="images\rulebased-chatbot_part2.png" style="margin: 0 auto; display: block;"><br/>
  <em>Hình 3.1. Rule-based Chatbot</em>
</p>

Các lĩnh vực ứng dụng của rule-based chatbot:
-	Chăm sóc khách hàng: Trả lời câu hỏi FAQs, báo cáo thông tin đặt hàng, đưa ra các gợi ý xử lí vấn đề đơn giản.
-	Y tế: Hỗ trợ đặt lịch khám, cung cấp thông tin y tế, thông tin khám bệnh, hồ sơ bệnh nhân.
-	Ngân hàng: Trả lời các câu hỏi cơ bản về giao dịch, các dịch vụ của ngân hàng.

Ưu điểm: 
-	Dễ thiết lập và sử dụng nhờ sử dụng lập trình quy tắc, có thể triển khai nhanh chóng mà không cần đầu tư huấn luyện AI.
-	Hiệu quả trong việc xử lí các tác vụ mang tính lặp lại, trả lời nhanh các câu hỏi thường gặp, tiết kiệm nhân lực cho các tác vụ này.
-	Có thể đưa ra phản hồi chính xác do được lập trình sẵn theo khuôn mẫu.
-	Chi phí phát triển và vận hành thấp.

Nhược điểm:
-	Không trả lời được các câu hỏi ngoài phạm vi đã được lập trình sẵn.
-	Không thể tự học, khó phát triển vì mỗi khi doanh nghiệp muốn them tính năng mới, phải cập nhật chatbot
-	Không thể xử lí đoạn hội thoại phức tạp, giảm trải nghiệm khách hàng.

Công cụ phát triển:
-	Vì dựa trên lập trình quy tắc, có thể lập trình rule-based chatbot bằng các ngôn ngữ lập trình như Python. Các điều kiện được triển khai theo cấu trúc if-else, pattern matching.


## Level 2: ML-based Chatbot

Machine Learning chatbot ứng dụng các thuật toán machine learning và NLP trong quá trình xây dựng chatbot. Khác với rule-based chatbot, ML-based chatbot đưa ra phản hồi thông minh và linh hoạt hơn nhờ vào quá trình đào tạo bằng AI thay vì lập trình quy tắc. 

<p align="center">
  <img src="images\ai-chatbot_part2.png" style="margin: 0 auto; display: block;"><br/>
  <em>Hình 3.2. Machine Learning-based Chatbot</em>
</p>

Ứng dụng của ML-based chatbot:
-	Giống với rule-based chatbot, ML-based chatbot được ứng dụng rộng rãi trong lĩnh vực chăm sóc khách hàng, y tế, dịch vụ. Tuy nhiên, với việc ứng dụng machine learning, chất lượng phản hồi của chatbot được nâng cấp hơn. 
-	Trong lĩnh vực chăm sóc khách hàng, dịch vụ, ngoài cung cấp thông tin, chatbot còn có thể gợi ý them thông tin dựa trên cuộc hội thoại trước đó.
-	Trong y tế, ngoài trả lời câu hỏi về lịch hẹn, thông tin sức khỏe, ML-based chatbot còn có thể theo dõi và báo cáo bác sĩ về tình trạng bệnh nhân để hỗ trợ kịp thời.

Ưu điểm:
-	Chatbot phản hồi linh hoạt hơn do có thể hiểu ngữ cảnh của người dung, cung cấp nhiều thông tin hơn, nâng cao trải nghiệm người dùng.
-	Chatbot có khả năng tự học từ dữ liệu và thông qua tương tác khách hang, từ đó hệ thống luôn được cập nhật tự động.

Nhược điểm:
-	Chi phí đào tạo và vận hành lớn. Việc ứng dụng AI đòi hỏi bộ dữ liệu lớn và chất lượng tùy lĩnh vực để đào tạo chatbot.
-	Thời gian đào tạo và triển khai chatbot cũng lâu hơn rule-based chatbot.

Công cụ phát triển:
Hiện nay có rất nhiều công cụ, thư viện để xây dựng AI chatbot, có thể kể đến như:
- Tensorflow, Pytorch: Hai thư viện và framework nổi tiếng về deep learning để đào tạo ML-based chatbot. Gồm các thuật toán, thư viện để hỗ trợ quá trình xây dựng và đào tạo chatbot.
- spaCy: Thư viện NLP để xử lí ngôn ngữ tự nhiên.
- Hugging Face Transformers: Nền tảng lưu trữ các pre-trained models lớn như GPT, BERT.
- Rasa: Framework mã nguồn mở lớn cho chatbot, bao gồm các kĩ thuật như NLU, intent classification, entity extraction.

## Level 3: LLM-based Chatbot

LLM-based chatbot có thể xem là một trợ lí ảo được vận hành bởi Large Language Model, được đào tạo trên bộ dữ liệu khổng lồ. LLM-based chatbot có khả năng hiểu ngôn ngữ, tạo phản hồi tự nhiên và tương tác giống người.

<p align="center">
  <img src="images\LLM_chatbot_part2.png" style="margin: 0 auto; display: block;"><br/>
  <em>Hình 3.3. LLM-based chatbot</em>
</p>

Ứng dụng của LLM-based chatbot:
-	Với LLM, chatbot có thể được ứng dụng như một trợ lí ảo cho nhiều lĩnh vực khác nhau. Có thể đóng vai trò như một trợ lí giải đáp thắc mắc về dịch vụ cho cơ sở y tế, dịch vụ.
-	Hỗ trợ giải đáp về chính sách, văn bản, tóm tắt và dịch tài liệu.
-	Hỗ trợ giải quyết vấn đề kĩ thuật theo hướng dẫn cụ thể.
-	Tạo nội dung dựa trên yêu cầu.

Ưu điểm:
-	Hiểu tường tận ngôn ngữ con người, có thể trả lời các câu hỏi phức tạp, đưa gợi ý và hướng dẫn, không rập khuôn, đa ngành nghề.
-	Xử lí được các yêu cầu phức tạp, tổng quát, có khả năng sáng tạo nội dung.
-	Tự động hóa các công việc như báo cáo, tổng hợp thông tin, giúp tiết kiệm thời gian và tang năng suất.

Nhược điểm:
-	Cần tài nguyên tính toán lớn. Việc huấn luyện và vận hành LLM yêu cầu phần cứng, cơ sở vật chất mạnh mẽ, tốn kém.
-	Vì được đào tạo trên mô hình ngôn ngữ lớn, không tránh khỏi các trường hợp huấn luyện sai về kiến thức, từ đó nội dung được tạo ra có thể sai.

Công cụ phát triển:
-	LangChain: framework mã nguồn mở hỗ trợ xây dựng chatbot dựa trên mô hình ngôn ngữ lớn.
-	Llama: mô hình ngôn ngữ lớn mã nguồn mở của Meta.
-	OpenAI API: mô hình AI của OpenAI, cho phép ứng dụng truy cập vào mô hình để tạo chatbot
-	Hugging Face Transformers: Nền tảng lưu trữ các pre-trained models lớn.

# 4. Bạn muốn tạo chatbot để làm gì?
# 4.1 Xác định mục tiêu của chatbot
Trên thực tế, phần lớn AI chatbot hiện nay có thể được xếp vào một trong bốn nhóm chính.

**FAQ Bot – Trả lời câu hỏi thường gặp**
Đây là dạng chatbot phổ biến nhất, thường dùng trong chăm sóc khách hàng.
- Trả lời các câu hỏi lặp lại: giờ làm việc, chính sách, hướng dẫn sử dụng
- Không cần hội thoại quá dài
- Nội dung tương đối cố định

Loại chatbot này phù hợp để giảm tải cho con người, đặc biệt trong các hệ thống hỗ trợ khách hàng.

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20260123_152134_f48bddef.png" style="margin: 0 auto; display: block;"><br/>
  <em>Hình 4.1. FAQ chatbot</em>
</p>

**Task-oriented Bot – Chatbot thực hiện tác vụ**
Khác với FAQ Bot, loại chatbot này không chỉ trả lời mà còn **dẫn người dùng qua một quy trình**.

Ví dụ:
- Đặt lịch hẹn
- Booking dịch vụ
- Tra cứu thông tin theo từng bước

Trọng tâm của chatbot dạng này là logic và luồng hội thoại, không phải kiểu nói chuyện tự nhiên.

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn//static/uploads/20260123_153156_3c398178.jpeg" style="margin: 0 auto; display: block;"><br/>
  <em>Hình 4.2.Task-oriented Bot</em>
</p>

**Conversational Bot – Chatbot trò chuyện tự nhiên**
Đây là dạng chatbot giống một “bạn trò chuyện”.
- Mục tiêu là duy trì hội thoại
- Câu trả lời cần tự nhiên, linh hoạt
- Không nhất thiết phải “đúng tuyệt đối”
  
Loại chatbot này thường được dùng cho giải trí, hỗ trợ tinh thần hoặc tương tác xã hội.

Tuy nhiên có một lưu ý: Conversational bot khó làm tốt hơn các loại chatbot khác, vì yêu cầu xử lý ngữ cảnh và lịch sử hội thoại dài.

![conservational bot](https://github.com/user-attachments/assets/9f5f7a38-1b2f-4e81-9f45-819263d6f07c)
<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn//static/uploads/20260123_153349_1ce0facd.webp" style="margin: 0 auto; display: block;"><br/>
  <em>Hình 4.3. Conversational Bot</em>
</p>

**Domain-specific Bot – Chatbot cho lĩnh vực cụ thể**

Chatbot được thiết kế cho một lĩnh vực nhất định như:

- Y tế
- Giáo dục
- Bán hàng

Đặc điểm của loại này:
- Cần dữ liệu riêng
- Phải kiểm soát chặt nội dung
- Sai sót có thể gây hậu quả lớn

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn//static/uploads/20260123_155702_32fc2bbb.jpeg" style="margin: 0 auto; display: block;"><br/>
  <em>Hình 4.4. Domain-specific Bot</em>
</p>

**Những câu hỏi bắt buộc phải trả lời trước khi code**

Sau khi xác định loại chatbot, bạn cần trả lời rõ các câu hỏi sau:
- Chatbot này dùng cho ai?
- Nó sẽ trả lời loại câu hỏi nào?
- Có cần nhớ lịch sử hội thoại hay chỉ trả lời từng câu riêng lẻ?
- Có cần dùng dữ liệu riêng không, hay chỉ kiến thức chung?

 Nếu chưa trả lời rõ những câu hỏi này, việc code sẽ rất dễ **“loạn hướng”** khiến tính năng thêm khó sửa, khó mở rộng.

# 4.2 Những sai lầm thường gặp khi bắt đầu tạo chatbot

Khi mới làm chatbot, rất nhiều người gặp những sai lầm giống nhau:

**Kỳ vọng chatbot “hiểu” như con người**

Chatbot không có nhận thức hay cảm xúc. Nó chỉ xử lý ngôn ngữ và dự đoán câu trả lời dựa trên dữ liệu đã học. Kỳ vọng chatbot suy nghĩ như con người sẽ dẫn đến thất vọng.

Ví dụ, bạn tạo một chatbot tư vấn sản phẩm và hỏi:

“Mình muốn mua điện thoại cho bố mẹ dùng cho tiện.”

 Con người sẽ tự hiểu rằng:

- Người lớn tuổi

- Ưu tiên dễ dùng, pin tốt, chữ to

Nhưng chatbot có thể chỉ bám vào từ khóa “điện thoại” và đưa ra danh sách sản phẩm phổ biến, không phù hợp với ngữ cảnh. Điều này xảy ra vì chatbot không có trải nghiệm sống hay suy luận xã hội, mà chỉ phân tích mẫu ngôn ngữ dựa trên dữ liệu đã học.

**Tin chatbot 100%**

AI chatbot có thể trả lời sai nhưng nghe rất thuyết phục. Nếu không có cơ chế kiểm soát, chatbot có thể tạo ra thông tin sai lệch mà người dùng khó nhận ra.

Ví dụ, một chatbot học tập được hỏi:

“Công thức này áp dụng trong trường hợp nào?”

Chatbot có thể trả lời rất chi tiết, dùng nhiều thuật ngữ chuyên môn, nhưng nội dung lại sai hoặc không còn đúng với kiến thức mới. Nếu người dùng không kiểm chứng lại, thông tin sai này có thể được sử dụng như sự thật.

Vấn đề ở đây không phải chatbot “nói dối”, mà là nó **không kiểm chứng thông tin**, chỉ dự đoán câu trả lời có khả năng xuất hiện cao nhất.

**Không giới hạn phạm vi**

Muốn chatbot “trả lời được mọi thứ” là một sai lầm phổ biến. Chatbot càng tập trung vào một phạm vi hẹp thì càng hiệu quả và dễ kiểm soát.

Ví dụ, bạn nói:

“Chatbot của mình vừa trả lời câu hỏi, vừa tư vấn, vừa trò chuyện, vừa làm trợ lý cá nhân.”

Kết quả thường là:

- Câu trả lời lan man
- Không rõ chatbot mạnh ở điểm nào
- Khó kiểm soát chất lượng
  
Trong thực tế, một chatbot chỉ trả lời tốt khi phạm vi nhiệm vụ của nó được giới hạn rõ ràng. Chatbot trả lời FAQ sẽ khác chatbot đặt lịch, và cả hai đều khác chatbot trò chuyện tự nhiên.

**Bỏ qua chi phí và bảo mật**

Khi mới làm chatbot, nhiều người chỉ tập trung vào việc “làm cho chạy được”, mà quên mất các vấn đề phía sau.

Ví dụ:

- Đặt API key trực tiếp trong code và upload lên GitHub
- Không giới hạn số lượng request
- Không theo dõi chi phí sử dụng API
  
Hậu quả có thể là:
- API key bị lộ
- Tài khoản bị sử dụng trái phép
- Chi phí tăng nhanh ngoài dự kiến
  
Những vấn đề này thường chỉ xuất hiện sau khi chatbot đã chạy, và lúc đó việc sửa lại sẽ tốn nhiều công sức hơn.

# 4.3 Khi nào nên bắt đầu làm demo?

Demo **không nên** là bước đầu tiên, mà là bước dùng để kiểm chứng xem ý tưởng của bạn có thực sự hiệu quả hay không.

Sau khi đã xác định rõ chatbot dùng để làm gì và phục vụ cho ai, lúc này bạn mới nên nghĩ đến việc làm một bản demo nhỏ. Demo nên được bắt đầu khi mục tiêu chatbot đã rõ ràng và phạm vi sử dụng đã được thu hẹp. Đây là thời điểm bạn cần kiểm tra một câu hỏi rất đơn giản nhưng quan trọng: *chatbot này có giải quyết đúng vấn đề mình đặt ra hay không?*

Một bản demo tốt không cần phải đầy đủ mọi tính năng, cũng không cần giao diện đẹp hay trải nghiệm hoàn hảo. Thay vào đó, demo chỉ cần tập trung vào phần cốt lõi nhất của chatbot. Nếu chatbot được tạo ra để trả lời câu hỏi, hãy kiểm tra xem nó có trả lời đúng và ổn định hay không. Nếu chatbot được thiết kế để hỗ trợ một tác vụ, hãy xem liệu nó có hoàn thành được tác vụ đó một cách trơn tru.

Mục tiêu của demo không phải là tạo ra sản phẩm hoàn chỉnh, mà là giúp bạn phát hiện sớm các vấn đề về ý tưởng, phạm vi hoặc cách tiếp cận. Một demo đơn giản nhưng đúng trọng tâm sẽ giúp bạn tiết kiệm rất nhiều thời gian và công sức khi bước sang giai đoạn phát triển chatbot đầy đủ hơn.

# 5. Xây dựng chatbot

Sau khi đã hiểu cách hoạt động và các thành phần của một AI chatbot, chúng ta sẽ làm một demo AI chatbot đơn giản chạy trực tiếp trên Google Colab:

Khác với cách tiếp cận phổ biến là gọi API từ các dịch vụ bên ngoài, trong blog này chatbot sẽ tải và chạy trực tiếp model AI trên môi trường Google Colab. Cách làm này giúp chúng ta hiểu rõ hơn cách mô hình hoạt động nội bộ, đồng thời phù hợp cho việc nghiên cứu, thử nghiệm và học tập mà không phụ thuộc vào API từ bên thứ ba.

## 5.1. Cài đặt thư viện cần thiết

Trước tiên, chúng ta cần cài đặt một số thư viện quan trọng để phục vụ cho việc tải và chạy mô hình ngôn ngữ trực tiếp trên Google Colab:

 - transformers: thư viện của Hugging Face, dùng để tải các mô hình ngôn ngữ lớn.

 - torch: framework nền tảng của deep learning, giúp thực hiện các phép tính tensor và huấn luyện mô hình.

 - accelerate: hỗ trợ tối ưu quá trình chạy mô hình, cấu hình CPU và GPU, phân bổ tài nguyên và tăng tốc suy luận mà không cần cấu hình phức tạp.

 - bitsandbytes: cho phép nạp mô hình ở dạng nén (8-bit hoặc 4-bit), giúp giảm đáng kể mức sử dụng với tài nguyên hạn chế.

```
!pip install -q -U torch transformers accelerate bitsandbytes
```

## 5.2. Tải mô hình ngôn ngữ
Trong demo này, chúng ta sử dụng model:

***Qwen2.5-1.5B-Instruct***

Đây là một mô hình:

- Nhẹ (~1.5B parameters)

- Được fine-tune cho hội thoại

- Phù hợp cho demo và chạy thử

Bạn cũng có tìm và thay mô hình phù hợp tại ***[Hugging Face](https://huggingface.co/models)***

```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


model_name = "Qwen/Qwen2.5-1.5B-Instruct"

#Nếu sử dụng GPU thì đặt thành True
use_gpu = False

print("⏳ Đang tải model ...")
if use_gpu==True:
    nf4_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        low_cpu_mem_usage =True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage =True
    )
tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
print("⏳ Đã tải và load model ...")
```
## 5.3 Hàm sử dụng chatbot đơn giản

Luồng xử lý của hàm này đúng với tư duy đã trình bày ở các phần trước:

- Nhận input từ người dùng

- Đóng gói input vào prompt

- Gửi prompt cho mô hình

- Nhận kết quả và in ra câu trả lời
```
def local_chatbot():
    user_input = input("\n👤 User: ")
    if user_input.lower() in ['bye', 'exit']: return
    
    promt = f"""<|im_start|>system
              Bạn là một trợ lý AI hữu ích. Trả lời ngắn gọn, đúng trọng tâm.
              <|im_end|>
              <|im_start|>user
              {user_input}
              <|im_end|>
              <|im_start|>assistant
            """
    # Tokenize
    inputs = tokenizer(promt, return_tensors="pt")
    
    # Generate
    outputs = model.generate(**inputs, max_new_tokens=200)
    
    # Decode
    response = tokenizer.decode(outputs[0])
    
    # Đoạn này cần xử lý chuỗi một chút để in ra cho đẹp
    print(f"🤖 Bot: {response.split("<|im_start|>assistant")[-1].strip().replace("<|im_end|>","")}")
    return response

response = local_chatbot()
```
<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn//static/uploads/20260123_155047_fe914bcb.png" style="margin: 0 auto; display: block;"><br/>
  <em>Hình 5.1. Kết quả </em>
</p>

***Full source code tại: [Google Colab](https://colab.research.google.com/drive/1vpn7lnZbX3niohOM_7jMayMYqrmBVlIT?usp=sharing)***

# 6. Kết luận: Tạo chatbot là bài toán tư duy trước khi là bài toán code

Qua bài viết này, có thể rút ra một điểm quan trọng:

***Xây dựng chatbot AI không bắt đầu từ code, mà bắt đầu từ tư duy thiết kế.***

Trước khi viết bất kỳ dòng code nào, bạn cần trả lời rõ ràng những câu hỏi nền tảng:

- Chatbot được tạo ra để giải quyết vấn đề gì?

- Ai là người sử dụng chính?

- Phạm vi câu hỏi và trả lời đến đâu?

- Có cần sử dụng dữ liệu riêng hay không?

Khi những câu hỏi này chưa được làm rõ, việc bắt tay vào code quá sớm thường dẫn đến:

- Hệ thống phức tạp nhưng hiệu quả thấp

- Chatbot trả lời lan man, khó kiểm soát chất lượng

- Tốn chi phí triển khai nhưng không giải quyết đúng nhu cầu thực tế

Ngược lại, khi tư duy đã rõ ràng:

- Việc lựa chọn công nghệ trở nên đơn giản và có mục đích

- Code chỉ còn là bước hiện thực hóa ý tưởng

- Hệ thống dễ mở rộng, dễ tối ưu và dễ bảo trì về lâu dài

*Trong bài blog tiếp theo, chúng ta sẽ đi từ demo đơn giản này để xây dựng một chatbot hoàn chỉnh hơn, sau đó deploy lên các nền tảng miễn phí để chạy như một sản phẩm demo thực sự.*
# Trích dẫn
Anthropic. (2025). Introducing Claude 4. https://www.anthropic.com/news/claude-4

AWS. (n.d.). What is Retrieval-Augmented Generation (RAG)? Amazon Web Services. https://aws.amazon.com/what-is/retrieval-augmented-generation

Google AI for Developers. (2026). Text generation | Gemini API. https://ai.google.dev/gemini-api/docs/text-generation

Hire A.I. Developers. (2025). Fine-tuning vs. from scratch: When to use the OpenAI API vs. building a custom LLM. https://hire-aidevelopers.com/blog/fine-tuning-llms-openai-api-vs-custom-llm

Hugging Face. (2026). Fine-tuning. https://huggingface.co/docs/transformers/en/training

Microsoft. (2025, February 13). 5 key features and benefits of retrieval augmented generation (RAG). Microsoft Cloud Blog. https://www.microsoft.com/en-us/microsoft-cloud/blog/2025/02/13/5-key-features-and-benefits-of-retrieval-augmented-generation-rag

OpenAI. (2025). OpenAI for developers in 2025. https://developers.openai.com/blog/openai-for-developers-2025

Rasiksuhail. (2026, January). The 2025 LLM API playbook: I tested all 4 major providers so you don't have to (Part 1/3: Choosing your stack). Medium. https://rasiksuhail.medium.com/the-2025-llm-api-playbook-i-tested-all-4-major-providers-so-you-dont-have-to-part-1-3-choosing-6dd11b47370b

ScienceDirect. (2025). A survey on chatbots and large language models: Testing and evaluation techniques. https://www.sciencedirect.com/science/article/pii/S2949719125000044
