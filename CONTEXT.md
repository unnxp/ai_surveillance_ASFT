# Smart Surveillance

ภาษากลางของระบบเฝ้าระวังพื้นที่อัจฉริยะ ใช้แยกสิ่งที่ระบบสังเกต สิ่งที่โมเดลเสนอ และสิ่งที่มนุษย์ยืนยันออกจากกันอย่างชัดเจน

## System Boundaries

**Camera**:
แหล่งภาพเฝ้าระวังที่มีตัวตนและตำแหน่งติดตั้งแน่นอน กล้องหนึ่งตัวอาจส่งภาพให้ Site Edge Gateway เดียวกันกับกล้องอื่นได้
_Avoid_: Edge Node, Camera Node

**Site Edge Gateway**:
ขอบเขตการประมวลผลประจำพื้นที่ซึ่งรับข้อมูลจากกล้องหนึ่งตัวหรือหลายตัว ก่อนส่งเฉพาะข้อมูลที่จำเป็นไปยัง Central Server
_Avoid_: Edge Camera, Edge Node เมื่อไม่ชัดว่าหมายถึงกล้องหรือ Gateway

**Central Server**:
ขอบเขตส่วนกลางที่รวมข้อมูลจากหนึ่งหรือหลายพื้นที่ เพื่อเชื่อมโยงบุคคลข้ามกล้อง จัดการเหตุการณ์ และประสานการแจ้งเตือน
_Avoid_: AI Server เมื่อไม่ระบุความรับผิดชอบ

## Observations and Identity

**Observation**:
การพบเห็นบุคคลหนึ่งครั้งจากกล้องหนึ่งตัว ณ เวลาหนึ่ง โดยยังไม่สรุปว่าเป็นใครหรือเกี่ยวข้องกับเหตุใด
_Avoid_: Detection เมื่อหมายถึงข้อมูลที่ผ่านการผูกกับเวลาและกล้องแล้ว

**Local Track**:
ลำดับ Observation ที่ระบบสันนิษฐานว่าเป็นบุคคลเดียวกันภายในกล้องและช่วงการติดตามเดียวกัน รหัสของ Local Track ไม่มีความหมายข้ามกล้องหรือข้ามการเริ่มระบบใหม่
_Avoid_: Person ID, Global ID

**Tracklet**:
ชุด Observation แบบมีขอบเขตเวลาของ Local Track ซึ่งใช้เป็นหลักฐานสำหรับการเชื่อมโยงบุคคลหรือเหตุการณ์
_Avoid_: Track เมื่อกล่าวถึงข้อมูลที่ส่งออกจากกล้องแล้ว

**Face Registry**:
ทะเบียนบุคคลที่สมัครหรือลงทะเบียนด้วยความยินยอมสำหรับการจดจำใบหน้า เช่น พนักงาน เจ้าหน้าที่ หรืออาสาสมัคร การอยู่ในทะเบียนไม่ได้หมายความว่าเป็นบุคคลต้องสงสัยหรือมีสิทธิ์เข้าทุกพื้นที่
_Avoid_: Watchlist, Blacklist

**Registered Identity**:
ตัวตนแบบมีชื่อหรือรหัสบุคคลที่อยู่ใน Face Registry และมีภาพหรือ Face Embedding เพียงพอสำหรับการเปรียบเทียบ การจดจำตัวตนได้ไม่เท่ากับการมีสิทธิ์เข้าพื้นที่
_Avoid_: Watchlist Identity, Authorized Person เมื่อยังไม่ได้ตรวจ Access Permission

**Access Permission**:
กฎที่เชื่อม Registered Identity หรือ Role กับ RestrictedZone ที่อนุญาตให้เข้า ใช้แยกผล `authorized`, `unauthorized` และ `identity unavailable` ออกจากผลการจดจำใบหน้า
_Avoid_: `is_authorized` แบบ global ที่ไม่ระบุพื้นที่หรือเงื่อนไข

**Possible Identity**:
Registered Identity ที่ระบบคาดว่าอาจตรงกับบุคคลที่พบ แต่หลักฐานยังไม่ถึงเกณฑ์ match ไม่ถือเป็นการรับรองตัวบุคคล
_Avoid_: Identified Person, Confirmed Identity

**Camera Topology**:
กราฟมีทิศระหว่างกล้อง ที่กำหนดว่าบุคคลเดินทางจากกล้องต้นทางไปกล้องปลายทางได้ภายในเวลาต่ำสุดและสูงสุดเท่าใด ใช้จำกัดการเชื่อมโยงข้ามกล้อง ไม่ใช่พิกัดภูมิศาสตร์
_Avoid_: Site Map, Camera Layout

**Global Person ID**:
รหัสนามแฝงที่ใช้เชื่อม Tracklet ซึ่งมีแนวโน้มเป็นบุคคลเดียวกันข้ามกล้อง โดยไม่ยืนยันชื่อจริงหรือความผิด
_Avoid_: Criminal ID, Face ID, Track ID

**Person of Interest**:
บุคคลที่ระบบหรือเจ้าหน้าที่เชื่อมโยงกับ Event Candidate หรือ Confirmed Incident และควรได้รับการตรวจสอบเพิ่มเติม บุคคลนี้อาจเป็น Registered Identity หรือบุคคล Unknown และไม่ได้เป็นสมาชิก Face Registry เพราะสถานะนี้โดยอัตโนมัติ คำนี้ไม่ใช่ข้อสรุปว่าบุคคลนั้นกระทำผิด
_Avoid_: Criminal, Offender, Culprit

## Events and Evidence

**Event Candidate**:
เหตุที่ระบบเสนอจาก Observation หรือ Tracklet เพื่อให้เจ้าหน้าที่ตรวจสอบ ยังไม่ถือว่าเป็นเหตุการณ์จริงที่ได้รับการยืนยัน
_Avoid_: Incident, Crime, Alarm Event

**Confirmed Incident**:
Event Candidate ที่เจ้าหน้าที่ตรวจสอบหลักฐานและยืนยันแล้ว โดยเจ้าหน้าที่อาจแก้ประเภทเหตุหรือข้อมูลบุคคลก่อนยืนยัน
_Avoid_: Detection, Prediction

**Incident**:
กลุ่มของ Event Candidate ที่ Central เชื่อมโยงกันจากบุคคล กล้อง และช่วงเวลาเดียวกันเป็นเส้นเวลาเดียว เพื่อให้เจ้าหน้าที่ตรวจเป็นเรื่องเดียว Incident ยังไม่ใช่ Confirmed Incident จนกว่าเจ้าหน้าที่จะยืนยัน
_Avoid_: Case, Crime

**Acknowledged**:
สถานะที่เจ้าหน้าที่รับทราบ Alert แล้ว แต่ยังไม่ได้ตัดสินว่าเป็นเหตุจริงหรือไม่
_Avoid_: Confirmed, Resolved

**Reclassified Incident**:
เหตุการณ์จริงที่ระบบระบุชนิดไม่ถูกต้อง และเจ้าหน้าที่แก้ประเภทเหตุให้ถูกต้อง
_Avoid_: Rejected Candidate

**Rejected Candidate**:
Event Candidate ที่เจ้าหน้าที่ตรวจสอบแล้วว่าไม่ควรถูกจัดเป็น Confirmed Incident
_Avoid_: False Positive เว้นแต่ใช้ในบริบทการประเมินโมเดล

**Evidence Package**:
ชุดหลักฐานและบริบทที่มากับ Event Candidate เพื่อให้เจ้าหน้าที่ตรวจสอบเหตุ บุคคล เวลา และกล้องต้นทางได้
_Avoid_: Screenshot เมื่อชุดหลักฐานมีข้อมูลมากกว่าภาพเดียว

**Alert**:
การนำ Event Candidate พร้อม Evidence Package ไปแจ้งให้เจ้าหน้าที่รับทราบ Alert ไม่ใช่การยืนยันเหตุหรือความผิดของบุคคล
_Avoid_: Incident, Conviction

**Alert Policy**:
กฎเชิงปฏิบัติการต่อชนิด Event Candidate ที่กำหนดการเปิดปิด severity, routing และ cooldown ไม่รวม ML confidence threshold
_Avoid_: Threshold, Model Setting

**Insufficient**:
ผลของโมเดลเมื่อ confidence หรือคุณภาพหลักฐานไม่พอจะสรุป ระบบงดตัดสินแทนการตีเป็นปกติ
_Avoid_: Normal

**Unavailable**:
ผลด้านตัวตนเมื่อระบุไม่ได้ เช่น ไม่เห็นหน้า ภาพไม่ผ่าน quality gate หรือโมเดล/Central ใช้ไม่ได้ แยกจาก Unknown ซึ่งหมายถึงตรวจแล้วไม่ตรงกับทะเบียน
_Avoid_: Unknown

## Spatial Rules

**RestrictedZone**:
พื้นที่ในภาพที่มีกฎการเข้าถึงหรือเงื่อนไขด้านความปลอดภัยเฉพาะ การอยู่ในพื้นที่นี้เป็นบริบทหนึ่งของเหตุ ไม่ใช่ข้อพิสูจน์การกระทำผิด
_Avoid_: Geofence, ROI

**InferenceROI**:
พื้นที่ในภาพที่อนุญาตหรือจัดลำดับความสำคัญให้การวิเคราะห์ที่ใช้ทรัพยากรสูง โดยไม่จำเป็นต้องเป็นพื้นที่หวงห้าม
_Avoid_: RestrictedZone, Geofence

**Tripwire**:
เส้นเสมือนที่ใช้บอกการข้าม ทิศทางเข้าออก หรือการเปลี่ยนผ่านระหว่างพื้นที่
_Avoid_: RestrictedZone, ROI
