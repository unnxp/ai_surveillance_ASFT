# AI_Based_Smart_Surveillance_System

ระบบเฝ้าระวังพื้นที่อัจฉริยะโดยใช้ปัญญาประดิษฐ์ — เป้าหมายและขอบเขต MVP อยู่ใน [ARCHITECTURE.md](ARCHITECTURE.md)

## สถานะการพัฒนา

Week 1A: Event Contract, Central FastAPI, SQLite/WAL, HTTP client, Dashboard ขั้นต่ำ และ Shared Gateway skeleton

Week 1B: ตัวรันสองวิดีโอ, อ่านเฟรมใหม่ทีละเฟรม, จำกัดอัตราประมวลผล, โหลด YOLO ร่วมกันหนึ่งตัว, ByteTrack แยกต่อกล้อง และการทดสอบ HTTP บน localhost ตั้งแต่ Gateway ถึง SQLite และ Dashboard

Week 2A: เปิดใช้ Action Classifier ที่เทรนไว้แบบเลือกได้ สะสม 32 เฟรมจริงต่อ track, จำกัดจำนวน active tracks, จัดคิวข้ามสองกล้อง และสร้าง Event Candidate หลังผ่าน temporal rule 2 ใน 3 windows

Week 2B: InferenceROI, RestrictedZone และ Tripwire แยกต่อกล้องใน `configs/spatial_rules.yaml`; Tripwire Candidate เข้าถึง Central/SQLite แล้ว และมีเครื่องมือวาดจากเฟรมวิดีโอท้องถิ่น

Week 2C: Central มี Face Registry และ Access Permission ขั้นพื้นฐานใน SQLite; เก็บสถานะความยินยอม, template ใบหน้าพร้อม model version และสิทธิ์ต่อคู่ `(camera_id, zone_id)`

Week 2D: เปิด `--face` เพื่อให้ Gateway สกัดใบหน้าด้วย SCRFD/ArcFace บน CPU จาก track ที่ผ่าน InferenceROI ส่ง embedding ไป Central จับคู่ทะเบียนและตรวจสิทธิ์ราย Zone แล้วสร้าง Event Candidate ระดับ HIGH แบบทดลองเมื่อผลใบหน้าที่ใช้ได้อย่างน้อยสองตัวอย่างตรงกัน

Week 2E: ภาพหลักฐานของ Event Candidate (Keyframe และ Person Crop JPEG) สกัดอัตโนมัติและจัดเก็บใต้ `data/evidence/` พร้อมปุ่มเปิดดูบน Dashboard

Week 3A: Camera Topology (Directed Graph พร้อมกรอบเวลา min/max travel time), Anonymous Global Person ID (`GP-0001` Session 2 นาที/120s), และ Cross-Camera Association Engine เชื่อมโยงบุคคลและ Event Candidates ข้ามกล้องบน Central Server

Week 3B: Central Incident Timeline (ร้อยเรียงประวัติเหตุการณ์ของบุคคลเดียวกันข้ามกล้องเป็น Incident เดียว) และ Alert Policy Engine (กำหนดระดับความรุนแรง, เส้นทางการแจ้งเตือน, และระบบกรอง Cooldown Deduplication ลดการแจ้งเตือนซ้ำซ้อน) พร้อม API จัดการนโยบาย

Week 3C: Human Review Workflow บน Dashboard (บันทึกการตรวจสอบของเจ้าหน้าที่: Acknowledge, Confirm พร้อม final severity, Reject พร้อม false-positive reason, Reclassify), จัดเก็บ Structured Review Records ในตาราง `incident_reviews` สำหรับ Audit Trail พร้อม UI จัดการบนหน้าเว็บ

เส้นทางใหม่อยู่ใน `edge/` และ `server/`; `main.py` และ `main_cam2.py` เป็นต้นแบบเดิม (legacy) ที่ยังเรียก `logic/risk_assessment.py` และ Heatmap ซึ่ง ARCHITECTURE.md นำออกจาก MVP แล้ว ไม่ใช่เส้นทางหลักของระบบ

### สรุปสิ่งที่ทำแล้ว / ยังไม่ทำ

| ส่วน | สถานะ |
|---|---|
| Shared Gateway สองวิดีโอ, YOLO ร่วม, ByteTrack ต่อกล้อง | ทำแล้ว (`edge/`) |
| Action Classifier + temporal rule 2/3 | ทำแล้ว (threshold 0.70 ยังไม่ calibrate) |
| InferenceROI / RestrictedZone / Tripwire | ทำแล้ว (วาดด้วย `tools.spatial_drawer`) |
| Face Registry + Access Permission + Face-Zone Candidate | ทำแล้ว (threshold 0.55 ยังไม่ผ่าน open-set validation) |
| Evidence keyframe + person crop | ทำแล้ว |
| Camera Topology, Global Person ID, Incident, Alert Policy, Human Review | ทำแล้วที่ Central (API); Dashboard มีเฉพาะ Incident review และดู Evidence |
| Person ReID | ยังไม่ทำ (association ใช้ face similarity + topology) |
| WebSocket, คลิป 3s/5s, RAM ring buffer, retention | ยังไม่ทำ |
| RTSP/USB ใน `edge.run_gateway` | ยังไม่ทำ (รับไฟล์วิดีโอและ synthetic) |
| Zone Editor / Face Registry / Policy / Topology UI บนเว็บ | ยังไม่ทำ (ใช้ API หรือ CLI แทน) |

## เปิด Central และ Dashboard

รันจาก repository root โดยใช้ dependency ที่มีอยู่ใน `env`:

```powershell
.\env\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

เปิด `http://127.0.0.1:8000/` เพื่อดู Dashboard และ `/docs` เพื่อดู API
SQLite จะถูกสร้างที่ `data/surveillance.db` ตอน Central เริ่มทำงานเท่านั้น ภาพ/วิดีโอไม่ได้ถูกส่งหรือเก็บในฐานข้อมูลนี้

## Face Registry ขั้นพื้นฐาน

Central เปิด `PUT /api/v1/face-registry/{person_id}` เพื่อเพิ่มหรือแทนข้อมูลบุคคล, `GET /api/v1/face-registry` เพื่อดูรายการ metadata และ `DELETE /api/v1/face-registry/{person_id}` เพื่อถอนความยินยอมและลบทะเบียนพร้อม template/สิทธิ์ที่เกี่ยวข้อง คำขอ PUT ต้องยืนยัน `consent_confirmed=true` และมี template อย่างน้อยหนึ่งชุด แต่ละชุดมี `model_version` และเวกเตอร์ 512 ค่าที่ normalize แล้ว; ผลตอบกลับและรายการทะเบียนไม่แสดงเวกเตอร์

สิทธิ์ใช้รายการ `allowed_zones` ต่อคน โดยระบุ `camera_id` และ `zone_id` คู่กัน: ชื่อ Zone เหมือนกันต่างกล้องไม่ถือเป็นสิทธิ์เดียวกัน `role` เป็นข้อมูลประกอบในขั้นนี้ ยังไม่มี role-wide permission การอยู่ในทะเบียนไม่ได้หมายความว่ามีสิทธิ์ทุกพื้นที่

มี `POST /api/v1/face-registry/match` สำหรับ embedding 512 ค่าแบบ normalize พร้อม `model_version`, `camera_id` และ `zone_ids`; Central ส่งกลับ Registered/Unknown/Unavailable กับสิทธิ์ราย Zone โดยไม่บันทึก query vector ลง SQLite หรือส่งกลับใน response ควรใช้ผ่าน Gateway บน localhost เท่านั้น

ลงทะเบียนจากภาพท้องถิ่นที่เจ้าตัวให้ความยินยอมแล้วได้ด้วย `tools.enroll_face` โดยต้องระบุ `--consent-confirmed` และภาพ 1–3 ภาพที่เห็นใบหน้าเดียวชัดเจน ตัวอย่างคำสั่ง (อย่ารันกับข้อมูลจริงโดยไม่มีความยินยอม):

```powershell
.\env\Scripts\python.exe -m tools.enroll_face --person-id employee_01 --name "Example Employee" --role staff --image "C:\consented\employee.jpg" --zone camera_01:storage --consent-confirmed
```

เครื่องมืออ่านภาพจากเครื่องและส่งเฉพาะ embedding ไป Central ใน localhost; API PUT จะเก็บ template พร้อม metadata และสิทธิ์ใน SQLite ไม่เก็บไฟล์ภาพต้นฉบับ ตัวอย่างนี้เป็นเพียงรูปแบบคำสั่ง ไม่ได้ใช้ภาพบุคคลจริงในการทดสอบรอบนี้

## ทดสอบเส้นทางระบบด้วยข้อมูลสังเคราะห์

เปิด terminal อีกหน้าหนึ่ง:

```powershell
.\env\Scripts\python.exe -m edge.run_gateway --synthetic --demo
```

คำสั่งนี้ใช้เฟรมสังเคราะห์สองแหล่ง และสร้าง Event ทดสอบหนึ่งรายการต่อกล้องเพื่อพิสูจน์การส่งข้อมูล ไม่ใช่ผลทำนายจาก AI
รายการมี `reason_codes=["synthetic_test_only"]` และ `attributes.synthetic=true`
Event ทดสอบใช้ชนิด `possible_loitering` เพื่อทดสอบ contract เท่านั้น และต้องไม่ใช้เป็นผลประเมินความแม่นยำ
เมื่อแหล่งภาพจบหรือหยุดตาม `--max-cycles` Gateway ส่งสถานะ OFFLINE ให้ Central

หากต้องการจำกัดจำนวนรอบ:

```powershell
.\env\Scripts\python.exe -m edge.run_gateway --synthetic --demo --max-cycles 10
```

## รับวิดีโอสองไฟล์

โหมดตรวจเส้นทางระบบกับไฟล์วิดีโอ:

```powershell
.\env\Scripts\python.exe -m edge.run_gateway --demo --camera "camera_01=C:\clips\camera1.mp4" --camera "camera_02=C:\clips\camera2.mp4"
```

โหมด Person Detection + ByteTrack ที่ใช้โมเดลร่วมกัน:

```powershell
.\env\Scripts\python.exe -m edge.run_gateway --camera "camera_01=C:\clips\camera1.mp4" --camera "camera_02=C:\clips\camera2.mp4" --weights ".\yolo26n.pt" --fps 10
```

เปิด Action Classifier เพิ่มจากโหมด detection (ตัวอย่างใช้ YOLOv8n baseline ที่มีอยู่ใน repository):

```powershell
.\env\Scripts\python.exe -m edge.run_gateway --camera "camera_01=C:\clips\camera1.mp4" --camera "camera_02=C:\clips\camera2.mp4" --weights ".\yolov8n.pt" --device cpu --action --action-device cpu --fps 10
```

`--action` โหลด `models/action_classifier_3cls.pth` และ metadata ในเครื่องเท่านั้น; ถ้าไม่มีหรือไม่ตรงกันจะหยุดก่อน inference ไม่ใช้โมเดลสุ่มแทน ปรับ sample rate, 32-frame window, threshold, cooldown และจำนวน tracks ได้ที่ `configs/action_runtime.yaml` ค่าเริ่มต้นรองรับ 8 action tracks รวม; กล้องเดียวใช้ได้ถึง 8 หากอีกกล้องว่าง และเมื่อเต็มจะคืนช่องจากกล้องที่ครองมากกว่าให้กล้องที่เพิ่งมี track ใหม่ Inference สูงสุด 2 tracks ต่อรอบ

เปิด Face-Zone Candidate เพิ่มได้ด้วย `--face` ในคำสั่งสองวิดีโอจริง (ใช้ร่วมกับ `--action` ได้) โดยต้องมี `models/det_10g.onnx` และ `models/w600k_r50.onnx` ในเครื่องก่อนเริ่ม ตัวอย่าง: เพิ่ม `--face` ต่อท้ายคำสั่งโหมด Person Detection ด้านบน และวาด RestrictedZone พร้อมลงทะเบียนบุคคลก่อน โหมด `--synthetic --demo` ไม่รองรับ `--face` ปรับ quality gate และความถี่การสกัดที่ `configs/face_runtime.yaml`; ค่าเริ่มต้นลองไม่เกิน 2 tracks ต่อเฟรมต่อกล้อง, 3 face samples ต่อ track และเว้น 5 เฟรมระหว่างครั้ง

Face Match ใช้ cosine threshold `0.55` จาก runtime เดิมเป็น **ค่าเริ่มต้นทดลองที่ยังไม่ผ่าน open-set validation** ค่า `0.3275` จากการทดลองเดิมเป็น closed-set heuristic จึงไม่ใช่เกณฑ์ยืนยัน Unknown ทั้งคู่ Event ที่เกิดเป็น Candidate ให้เจ้าหน้าที่ตรวจ ไม่ใช่ข้อสรุปว่าบุคคลก่อเหตุ Dashboard แสดง `HIGH` และ `Experimental` แต่ยังไม่มีการส่งแจ้งเตือนภายนอก; ถ้าไม่เห็นหน้า ภาพไม่ผ่าน quality gate, Gallery ว่าง, model version ไม่ตรงหรือ Central ใช้ไม่ได้ ระบบจะ abstain เป็น `unavailable` ไม่แจ้ง Unknown

## ภาพหลักฐานของ Event Candidate (Week 2E)

เมื่อ Gateway ทำงานกับวิดีโอสองไฟล์ในโหมดจริง (ไม่ใช่ `--demo`) และเกิด Event Candidate ระบบจะสร้าง JPEG ภาพเต็มเฟรมกับภาพตัดเฉพาะบุคคลจาก track ที่เกี่ยวข้อง ส่ง metadata ก่อน แล้วอัปโหลดภาพแยกไป Central ภาพถูกจำกัดขนาดและเก็บเป็นไฟล์ใต้ `data/evidence/`; SQLite เก็บเฉพาะ metadata กับตำแหน่งไฟล์ Dashboard มีปุ่ม **Evidence** ในรายการเหตุการณ์เพื่อเปิดดูภาพ หากภาพสร้างหรือส่งไม่สำเร็จ เหตุการณ์ยังคงอยู่และแสดงว่าหลักฐานไม่ครบ

`--synthetic --demo` และ `--demo` ไม่สร้างภาพหลักฐาน เพราะ Candidate เป็นข้อมูลทดสอบ ไม่ใช่ผล AI ขณะนี้ยังไม่มีคลิป 3 วินาทีก่อน/5 วินาทีหลังเหตุการณ์, ระบบลบตาม retention หรือการส่งแจ้งเตือนภายนอก (การ Review Incident บน Dashboard มีแล้วตาม Week 3C) การรันกับวิดีโอที่ระบุตัวบุคคลได้ต้องใช้ข้อมูลที่ได้รับอนุญาตและได้รับอนุมัติตาม `AGENTS.md`

## วาดกฎพื้นที่แยกตามกล้อง

ใช้เฟรมแรกของไฟล์วิดีโอท้องถิ่นเป็นภาพอ้างอิง แล้วคลิกจุดด้วยเมาส์ กด Enter เพื่อบันทึก, Backspace เพื่อถอนจุดล่าสุด หรือ Esc เพื่อยกเลิก:

```powershell
.\env\Scripts\python.exe -m tools.spatial_drawer --camera "camera_01=C:\clips\camera1.mp4" --kind roi --name entrance
.\env\Scripts\python.exe -m tools.spatial_drawer --camera "camera_01=C:\clips\camera1.mp4" --kind zone --name storage
.\env\Scripts\python.exe -m tools.spatial_drawer --camera "camera_01=C:\clips\camera1.mp4" --kind tripwire --name doorway --direction both
```

ROI/Zone ใช้จุดอย่างน้อย 3 จุด; Tripwire ใช้ 2 จุด แต่ละรายการบันทึกแยกตามชื่อและกล้องเป็นพิกัด normalized 0–1 ใน `configs/spatial_rules.yaml` ชื่อซ้ำในชนิดเดียวกันจะไม่เขียนทับ หากไม่ได้วาด ROI จะใช้ทั้งเฟรม; หากไม่มี Zone หรือ Tripwire จะไม่สร้าง Candidate ชนิดนั้น ระบุไฟล์กฎอื่นด้วย `--spatial-config` ได้

`--direction` ของ Tripwire เลือก `both`, `negative_to_positive` หรือ `positive_to_negative` โดยเครื่องหมายเป็นด้านของเส้นที่ลากจากจุดแรกไปจุดที่สอง กลไกใช้ footpoint ของ track ที่ต่อเนื่องอย่างน้อย 3 เฟรมและต้องข้ามช่วงเส้นจริง ไม่ใช่เส้นที่ยืดออกไปไม่สิ้นสุด

InferenceROI กรองเฉพาะงาน downstream เช่น Action และ Face ไม่ได้เพิ่มรอบ YOLO หรือหยุด Tracking ส่วน RestrictedZone จะสร้าง Candidate เฉพาะเมื่อมีผล face identity/access ที่ประเมินแล้วอย่างชัดเจนอย่างน้อยสองตัวอย่างและตำแหน่งอยู่ในพื้นที่ต่อเนื่อง การพบคนเพียงอย่างเดียวหรือไม่เห็นหน้าไม่สร้าง Unknown/Unauthorized Zone Candidate

Action ใช้เวลาในไฟล์วิดีโอตาม FPS ต้นฉบับในการเก็บตัวอย่าง ไม่เอาเฟรมเก่ามาเติมให้ครบ 32 เฟรม และแยก track ID ตามกล้อง ส่งเฉพาะ metadata/probabilities ไป Central ไม่ส่งภาพ ปัจจุบันเกณฑ์ 0.70 เป็น **baseline ที่ยังไม่ผ่าน calibration** และการใช้ person crop แตกต่างจากข้อมูลฝึกซึ่งเป็นภาพทั้งเฟรม จึงยังอ้างความแม่นยำของ deployed pipeline ไม่ได้ ต้องประเมินกับวิดีโอทดสอบที่เหมาะสมภายหลัง

YOLO26n เป็นค่าเริ่มต้น (`--weights` default คือ `yolo26n.pt` ที่ root) ต้องมี checkpoint `.pt` อยู่ในเครื่องก่อนเริ่ม ไม่มีการดาวน์โหลด weights อัตโนมัติ ปัจจุบัน repository มีเฉพาะ `yolov8n.pt` จึงต้องวาง `yolo26n.pt` เองหรือระบุ `--weights .\yolov8n.pt`
สามารถระบุ `--weights .\yolov8n.pt` เพื่อใช้ baseline ที่มีอยู่ หรือ `--device cpu` เพื่อใช้ CPU
ความเข้ากันได้และประสิทธิภาพของ YOLO26n ยังต้องทดสอบกับน้ำหนักจริง

การพบคนทั่วไปเป็น observation จึงยังไม่สร้าง Event ในโหมด detection; Dashboard แสดงสถานะกล้อง ส่วน tracks เก็บแยกใน Gateway
ถ้าไม่ใส่ `--action` จะไม่มี Action Candidate แต่ Tripwire ยังทำงานได้เมื่อกำหนดเส้นไว้
ไม่มี preview หรือวิดีโอส่งไป Central ในรอบนี้

ค่าจาก `configs/resource_limits.yaml` ที่ใช้แล้วคือ `max_fps_per_camera`, `yolo_device`, `yolo_half`, `action_device` เมื่อเปิด Action; ยังไม่บังคับ VRAM budget
`--fps` กำหนดอัตราอ่าน/ประมวลผลสูงสุดต่อกล้องในการ playback ไม่ใช่ benchmark FPS; Action ใช้ timestamp ที่คำนวณจาก FPS ของไฟล์เพื่อรักษา temporal sampling
ทั้งสองกล้องอ่านแบบ round-robin ใน process เดียว; live capture workers และการรับ RTSP/USB ยังไม่ได้เชื่อมในตัวรันนี้
EOF ไม่ย้อนกลับไปต้นไฟล์และไม่ประมวลผลเฟรมเก่าซ้ำ

## Topology, Incident, Alert Policy และ Review (Week 3A–3C)

API ทั้งหมดอยู่ใต้ `/api/v1` ของ Central บน localhost:

| Endpoint | หน้าที่ |
|---|---|
| `GET` / `PUT /topology` | ดูหรือแทนที่ edges ทั้งชุด แต่ละ edge มี `source_camera`, `destination_camera`, `min_travel_seconds` (ค่าเริ่มต้น 1), `max_travel_seconds` (ค่าเริ่มต้น 60), `enabled`; edge เป็นแบบมีทิศจาก source ไป destination |
| `GET /global-person-sessions?status=` | ดู Anonymous Global Person sessions (`GP-0001`…) ซึ่งหมดอายุหลัง 120 วินาที |
| `GET` / `PUT /alert-policy`, `POST /alert-policy/reset` | ดู แก้ หรือคืนค่าเริ่มต้นของ rule ต่อ Candidate type (`enabled`, `severity`, `routing`, `cooldown_seconds` ค่าเริ่มต้น 15) |
| `GET /incidents?status=`, `GET /incidents/{id}` | ดู Incident และ timeline |
| `POST /incidents/{id}/review` | บันทึก `action` (Acknowledge/Confirm/Reject/Reclassify) พร้อม `final_severity`, `reclassified_type`, `false_positive_reason` ลงตาราง `incident_reviews` |

ยังไม่มีหน้าเว็บสำหรับแก้ Topology หรือ Alert Policy ให้ใช้ `/docs` หรือส่ง JSON ตรง Policy Editor ไม่แก้ ML thresholds

## ตรวจสอบ

```powershell
.\env\Scripts\python.exe -m pytest -o addopts= tests -q
```

`-o addopts=` ใช้ปิด `--cov` ที่ตั้งไว้ใน `pyproject.toml` เพื่อให้รันได้โดยไม่วัด coverage และไม่ต้องพึ่ง pytest-cov
ชุด HTTP สร้างวิดีโอรูปทรง/สีสังเคราะห์และฐานข้อมูลชั่วคราว เปิดพอร์ตสุ่มเฉพาะ `127.0.0.1` และปิด server เมื่อทดสอบเสร็จ
ชุดตรวจ Week 2A โหลด Action checkpoint ที่มีอยู่ในเครื่องหนึ่งครั้งกับเฟรมสังเคราะห์เพื่อทดสอบ shape/probabilities ไม่ใช่การวัดความแม่นยำ ไม่ดาวน์โหลดโมเดล ไม่เข้ากล้องจริง และไม่ใช้ภาพบุคคล
ชุดตรวจ Week 2B ใช้รูปร่าง/วิดีโอสังเคราะห์และ UI mouse events ที่จำลองขึ้น จึงยังไม่ได้ยืนยันการใช้งานหน้าต่างวาดจริงบนเครื่องผู้ใช้

ข้อมูลสถานะเป็น heartbeat ล่าสุดที่ Central ได้รับ; การตรวจหมดเวลาเมื่อ Gateway crash จะเพิ่มภายหลัง
การส่ง HTTP ยังไม่มี retry/outbox หากส่งไม่สำเร็จ Gateway รายงาน failure และทำกล้องอื่นต่อ; ไม่รับประกันการส่ง Event ครบเมื่อเครือข่ายมีปัญหา
