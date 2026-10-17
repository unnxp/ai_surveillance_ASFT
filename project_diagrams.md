# ไดอะแกรมสรุปโครงสร้างโครงการ (Project Diagrams)
**โครงงาน:** ระบบเฝ้าระวังพื้นที่อัจฉริยะโดยใช้ปัญญาประดิษฐ์ (AI-Based Smart Surveillance System)

คุณสามารถนำรหัส **Mermaid** ด้านล่างนี้ไปวางในโปรแกรมวาดไดอะแกรม เช่น [Mermaid Live Editor](https://mermaid.live/) หรือเปิดผ่านโปรแกรมที่รองรับเพื่อบันทึกเป็นไฟล์ภาพนำไปใส่ในสไลด์พรีเซนต์และเล่มรายงานได้ทันทีครับ

---

## 🗺️ 1. สถาปัตยกรรมภาพรวมระบบ (Overall System Architecture)
แสดงการทำงานแบบกระจายศูนย์ (Distributed Edge-Central System) ระหว่างจุดติดตั้งกล้อง (Edge Nodes) และเซิร์ฟเวอร์หลักส่วนกลาง (Central Server)

```mermaid
graph TD
    %% Define styles
    classDef edgeNode fill:#d1ecf1,stroke:#17a2b8,stroke-width:2px,color:#0f5132;
    classDef server fill:#f8d7da,stroke:#dc3545,stroke-width:2px,color:#842029;
    classDef data fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#664d03;

    %% Edge Nodes
    subgraph Edge_Devices ["ฝั่งพื้นที่ติดตั้งกล้อง (Edge Node)"]
        Cam1[IP Camera 1] -->|RTSP/Video Stream| Edge1(Local Edge Node 1: MSI Laptop/Jetson)
        Cam2[IP Camera 2] -->|RTSP/Video Stream| Edge2(Local Edge Node 2: MSI Laptop/Jetson)
        
        Edge1 -->|Local Inference & Tracking| Proc1[YOLOv8 + ByteTrack<br>SCRFD + ArcFace<br>Action GRU]
        Edge2 -->|Local Inference & Tracking| Proc2[YOLOv8 + ByteTrack<br>SCRFD + ArcFace<br>Action GRU]
    end

    %% Network Transmission
    Proc1 -->|JSON Packet < 4KB| Net[Network / Local LAN]
    Proc2 -->|JSON Packet < 4KB| Net

    %% Central Server
    subgraph Central_Infrastructure ["ฝั่งเซิร์ฟเวอร์ควบคุมกลาง (Central AI Server)"]
        Net -->|HTTP POST / WebSockets| Fast[FastAPI Router]
        
        Fast -->|Extract Face Vector| FaceSearch[1:N Face Database Search<br>Cosine Similarity Matching]
        Fast -->|Extract Behavior Features| RiskMLP[Risk Assessment MLP Classifier<br>Predicts Low/Med/High/Critical]
        
        FaceSearch --> RiskMLP
        
        RiskMLP -->|Output Action & Risk Level| AlertMgr[Alert & Event Manager]
        AlertMgr -->|Log Event| CSV[(CSV / SQLite Event Log)]
        AlertMgr -->|Real-time UI Update| Dash[Web Dashboard / Live Alerts]
    end

    class Edge1,Edge2,Proc1,Proc2 edgeNode;
    class Fast,FaceSearch,RiskMLP,AlertMgr server;
    class CSV,Dash,Net data;
```

---

## 🔄 2. ลำดับการประมวลผลภายใน Edge Node (Edge Processing Pipeline)
แสดงวงจรการทำงาน (Pipeline) ของภาพวิดีโอสดในแต่ละเฟรม ตั้งแต่รับภาพดิบจากกล้อง จนถึงแพ็กส่งออกข้อมูลเป็น JSON

```mermaid
flowchart TD
    classDef startEnd fill:#e2e3e5,stroke:#383d41,stroke-width:2px;
    classDef step fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef decision fill:#f8d7da,stroke:#dc3545,stroke-width:2px;

    Start([รับเฟรมภาพดิบจากกล้อง BGR Frame]) --> Det[ตรวจจับวัตถุ YOLOv8 Detection]
    Det --> Track[ติดตามวัตถุ ByteTrack Association<br>กำหนด Track ID และกรอบพิกัด bbox]
    
    Track --> Crop[ครอปภาพบุคคล Bounding Box Crop]
    
    %% Face Recognition Path
    Crop --> FaceDet{พบใบหน้าในกรอบ?<br>SCRFD Face Detection}
    FaceDet -->|Yes| Align[ทำ Face Alignment 112x112]
    Align --> Embed[สกัดรหัสใบหน้า ArcFace Embedder<br>512-dim Vector]
    FaceDet -->|No| NoFace[กำหนด Face Vector = null]

    %% Action Classification Path
    Crop --> ActionInference{ถึงรอบวิเคราะห์พฤติกรรม?<br>Periodic Inference ~2s}
    ActionInference -->|Yes| ActionModel[วิเคราะห์ลำดับเฟรมเชิงเวลา<br>EfficientNet-B0 + Temporal GRU]
    ActionModel --> ActResult[ส่งคลาสพฤติกรรม 0/1/2<br>Normal/Property/Violence]
    ActionInference -->|No| KeepAction[ใช้ค่าผลลัพธ์ Action คลาสเดิม]

    %% Logic Verification Path
    Track --> Geofence[ตรวจสอบ Geofencing Polygons<br>เช็คเท้า/เข่า/สะเอว ล้ำเส้น]
    Track --> Loitering[นับระยะเวลาสะสมในพื้นที่หวงห้าม<br>Loitering Duration]
    Track --> Speed[คำนวณความเร็วเคลื่อนที่ Kalman Filter<br>Speed px/s]
    
    %% Integration and JSON Export
    Embed & NoFace & ActResult & KeepAction & Geofence & Loitering & Speed --> JSONPack[รวบรวมข้อมูลลง JSON Exporter Packet]
    JSONPack --> Export([ส่งออกข้อมูล JSON ไปยัง Central Server])

    class Start,Export startEnd;
    class Det,Track,Crop,Align,Embed,ActionModel,Geofence,Loitering,Speed,JSONPack step;
    class FaceDet,ActionInference decision;
```

---

## 🧠 3. การประมวลผลข้อมูลส่วนกลางและการจัดการความเสี่ยง (Central Risk Assessment Flow)
แสดงขั้นตอนที่เซิร์ฟเวอร์ส่วนกลางวิเคราะห์ระดับความเสี่ยงด้วยโมเดล MLP และการคัดกรองบุคคล

```mermaid
flowchart TD
    classDef input fill:#cce5ff,stroke:#004085,stroke-width:2px;
    classDef model fill:#e83e8c,stroke:#b1105c,stroke-width:2px,color:white;
    classDef action fill:#d4edda,stroke:#28a745,stroke-width:2px;

    JSON[รับข้อมูล JSON จาก Edge Node] --> Parse[แกะข้อมูลและแบ่งฟีเจอร์พฤติกรรม]
    
    %% Face Verification Branch
    Parse --> FaceEmbed[เวกเตอร์ใบหน้า 512 มิติ]
    FaceEmbed --> Match{เปรียบเทียบในระบบ 1:N Search<br>Cosine Similarity >= Threshold?}
    Match -->|Yes| Auth[ระบุชื่อบุคคล<br>Authorized Status = 1]
    Match -->|No| Stranger[ระบุเป็นคนแปลกหน้า<br>Stranger Status = 0]
    
    %% Feature Construction for MLP
    Parse --> Behav[ฟีเจอร์พฤติกรรมและพื้นที่ควบคุม]
    
    %% Risk Classifier Input (6 Dimensions)
    Auth & Stranger --> Construct[สร้างเวกเตอร์พฤติกรรม 6 มิติ:<br>1. Geofence Breach<br>2. Off-hours Status<br>3. Loitering Duration<br>4. Moving Speed<br>5. Authorized Status<br>6. Action Class / Confidence]
    Behav --> Construct
    
    Construct --> MLP[โมเดลประเมินความเสี่ยง<br>Feed-forward MLP Classifier]
    
    MLP --> Classify{ทำนายระดับความเสี่ยง<br>Risk Level}
    
    Classify -->|LOW / MEDIUM| Log[บันทึกข้อมูลประวัติเหตุการณ์ปกติลงฐานข้อมูล]
    Classify -->|HIGH / CRITICAL| Alert[เรียกใช้ Alert Manager เพื่อเตือนภัยทันที]
    
    Alert --> Capture[จับภาพเฟรมหลักฐานฝั่ง Edge]
    Alert --> WebPush[ส่งการแจ้งเตือน Real-time บน Dashboard]
    Alert --> Line[ส่งข้อความแจ้งเตือนทาง External API (Line/Telegram)]
    Alert --> CSV[บันทึกข้อมูลความเสียหายลง Event Log CSV]

    class JSON,FaceEmbed,Behav,Construct input;
    class MLP model;
    class Log,Alert,Capture,WebPush,Line,CSV action;
```
